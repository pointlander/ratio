// Copyright 2025 The Ratio Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bytes"
	"compress/bzip2"
	"embed"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"strconv"
	"strings"

	"github.com/pointlander/ratio/kmeans"
	order "github.com/pointlander/ratio/order24"
	"github.com/pointlander/ratio/order4"

	"github.com/pointlander/gradient/tf32"
	"github.com/pointlander/gradient/tf64"
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = 1.0e-2
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

//go:embed iris.zip
var Iris embed.FS

// Fisher is the fisher iris data
type Fisher struct {
	Measures []float64
	Label    string
	Cluster  int
	Index    int
}

// Labels maps iris labels to ints
var Labels = map[string]int{
	"Iris-setosa":     0,
	"Iris-versicolor": 1,
	"Iris-virginica":  2,
}

// Inverse is the labels inverse map
var Inverse = [3]string{
	"Iris-setosa",
	"Iris-versicolor",
	"Iris-virginica",
}

// Load loads the iris data set
func Load() []Fisher {
	file, err := Iris.Open("iris.zip")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		panic(err)
	}

	fisher := make([]Fisher, 0, 8)
	reader, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		panic(err)
	}
	for _, f := range reader.File {
		if f.Name == "iris.data" {
			iris, err := f.Open()
			if err != nil {
				panic(err)
			}
			reader := csv.NewReader(iris)
			data, err := reader.ReadAll()
			if err != nil {
				panic(err)
			}
			for i, item := range data {
				record := Fisher{
					Measures: make([]float64, 4),
					Label:    item[4],
					Index:    i,
				}
				for ii := range item[:4] {
					f, err := strconv.ParseFloat(item[ii], 64)
					if err != nil {
						panic(err)
					}
					record.Measures[ii] = f
				}
				fisher = append(fisher, record)
			}
			iris.Close()
		}
	}
	return fisher
}

//go:embed books/*
var Text embed.FS

var (
	// FlagCluster cluster mode
	FlagCluster = flag.Bool("c", false, "cluster mode")
	// FlagLM langauge model mode
	FlagLM = flag.Bool("lm", false, "language model mode")
	// FlagGA genetic algorithm language model mode
	FlagGA = flag.Bool("ga", false, "genetic algorithm language model mode")
)

// ClusterMode
func ClusterMode() {
	rng := rand.New(rand.NewSource(1))
	iris := Load()
	others := tf64.NewSet()
	others.Add("x", 4, len(iris))
	x := others.ByName["x"]
	for i := range iris {
		for _, value := range iris[i].Measures {
			x.X = append(x.X, value)
		}
	}

	set := tf64.NewSet()
	set.Add("y", 4, len(iris))
	for i := range set.Weights {
		w := set.Weights[i]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]float64, StateTotal)
			for ii := range w.States {
				w.States[ii] = make([]float64, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for range cap(w.X) {
			w.X = append(w.X, rng.NormFloat64()*factor)
		}
		w.States = make([][]float64, StateTotal)
		for ii := range w.States {
			w.States[ii] = make([]float64, len(w.X))
		}
	}

	drop := .7
	dropout := map[string]interface{}{
		"rng":  rng,
		"drop": &drop,
	}

	for iteration := range 1024 {
		pow := func(x float64) float64 {
			y := math.Pow(x, float64(iteration+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return y
		}
		sum := tf64.Add(others.Get("x"), set.Get("y"))
		l1 := tf64.T(tf64.Mul(tf64.Dropout(tf64.Mul(sum, sum), dropout), tf64.T(sum)))
		loss := tf64.Avg(tf64.Quadratic(l1, set.Get("y")))

		l := 0.0
		set.Zero()
		others.Zero()
		l = tf64.Gradient(loss).X[0]
		if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
			fmt.Println(iteration, l)
			return
		}
		fmt.Println(l)

		norm := 0.0
		for _, p := range set.Weights {
			for _, d := range p.D {
				norm += d * d
			}
		}
		norm = math.Sqrt(norm)
		b1, b2 := pow(B1), pow(B2)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}
		for _, w := range set.Weights {
			for ii, d := range w.D {
				g := d * scaling
				m := B1*w.States[StateM][ii] + (1-B1)*g
				v := B2*w.States[StateV][ii] + (1-B2)*g*g
				w.States[StateM][ii] = m
				w.States[StateV][ii] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				if vhat < 0 {
					vhat = 0
				}
				w.X[ii] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
			}
		}
	}
	embedding := make([][]float64, 150)
	y := set.ByName["y"]
	for i := range len(iris) {
		for ii := range 4 {
			value := y.X[i*4+ii]
			fmt.Printf("%f ", value)
			embedding[i] = append(embedding[i], value)
		}
		fmt.Println()
	}
	meta := make([][]float64, len(iris))
	for i := range meta {
		meta[i] = make([]float64, len(iris))
	}
	const k = 3
	for i := 0; i < 33; i++ {
		clusters, _, err := kmeans.Kmeans(int64(i+1), embedding, k, kmeans.SquaredEuclideanDistance, -1)
		if err != nil {
			panic(err)
		}
		for i := 0; i < len(meta); i++ {
			target := clusters[i]
			for j, v := range clusters {
				if v == target {
					meta[i][j]++
				}
			}
		}
	}
	clusters, _, err := kmeans.Kmeans(1, meta, k, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	for i, value := range clusters {
		iris[i].Cluster = value
	}

	sort.Slice(iris, func(i, j int) bool {
		return iris[i].Cluster < iris[j].Cluster
	})
	for _, value := range iris {
		fmt.Println(value.Cluster, value.Label)
	}

	a := make(map[string][3]int)
	for i := range iris {
		histogram := a[iris[i].Label]
		histogram[iris[i].Cluster]++
		a[iris[i].Label] = histogram
	}
	for k, v := range a {
		fmt.Println(k, v)
	}
}

// LMMode language model mode
func LMMode() {
	rng := rand.New(rand.NewSource(1))

	const (
		size  = 256
		step  = 16
		model = 8
		Eta   = 1.0e-3
	)

	const (
		// S is the scaling factor for the softmax
		S = 1.0 - 1e-300
	)

	softmax := func(values []float32) {
		max := float32(0.0)
		for i, v := range values {
			if v < 0 {
				v = -v
			}
			values[i] = v
		}
		for _, v := range values {
			if v > max {
				max = v
			}
		}
		s := max * S
		sum := float32(0.0)
		for j, value := range values {
			values[j] = float32(math.Exp(float64(value - s)))
			sum += values[j]
		}
		for j, value := range values {
			values[j] = value / sum
		}
	}

	type File struct {
		Name  string
		Data  []byte
		Model order4.Model
	}

	files := []File{
		{Name: "pg74.txt.bz2"},
		{Name: "10.txt.utf-8.bz2"},
		{Name: "76.txt.utf-8.bz2"},
		{Name: "84.txt.utf-8.bz2"},
		{Name: "100.txt.utf-8.bz2"},
		{Name: "1837.txt.utf-8.bz2"},
		{Name: "2701.txt.utf-8.bz2"},
		{Name: "3176.txt.utf-8.bz2"},
		{Name: "all"},
	}
	files[len(files)-1].Model.Init()

	load := func(book *File, all *File) {
		path := fmt.Sprintf("books/%s", book.Name)
		file, err := Text.Open(path)
		if err != nil {
			panic(err)
		}
		defer file.Close()
		breader := bzip2.NewReader(file)
		data, err := io.ReadAll(breader)
		if err != nil {
			panic(err)
		}

		mark := order4.Markov{}
		book.Model.Init()
		for _, value := range data {
			for ii := range mark {
				vector := book.Model[ii][mark[ii]]
				if vector == nil {
					vector = make([]uint32, size)
				}
				vector[value]++
				book.Model[ii][mark[ii]] = vector

				{
					vector := all.Model[ii][mark[ii]]
					if vector == nil {
						vector = make([]uint32, size)
					}
					vector[value]++
					all.Model[ii][mark[ii]] = vector
				}

				state := value
				for iii, value := range mark[ii][:ii+1] {
					mark[ii][iii], state = state, value
				}
			}
		}
		book.Data = data
	}
	for i := range files[:len(files)-1] {
		load(&files[i], &files[len(files)-1])
		fmt.Println(files[i].Name)
		for ii := range files[i].Model {
			fmt.Println(len(files[i].Model[ii]))
		}
	}

	str := []byte("What is the meaning of life?")
	process := func(str []byte) []byte {
		type String struct {
			String  []byte
			Entropy float64
		}
		results := make([]String, 256)
		done := make(chan bool, 8)
		process := func(seed int64, i int, str []byte) {
			rng := rand.New(rand.NewSource(seed))
			cp := make([]byte, len(str))
			copy(cp, str)
			length := len(cp) + step
			others := tf32.NewSet()
			others.Add("x", 256, length)
			x := others.ByName["x"]

			set := tf32.NewSet()
			set.Add("y", 256, length)

			var markov order4.Markov
			for _, value := range cp {
				markov.Iterate(value)
				distribution := files[model].Model.Lookup(&markov)
				for _, value := range distribution {
					x.X = append(x.X, value)
				}
			}
			for range step {
				distribution := files[model].Model.Lookup(&markov)
				sum, selected := float32(0.0), rng.Float32()
				for key, value := range distribution {
					sum += value
					if selected < sum {
						cp = append(cp, byte(key))
						markov.Iterate(byte(key))
						distribution := files[model].Model.Lookup(&markov)
						for _, value := range distribution {
							x.X = append(x.X, value)
						}
						break
					}
				}
			}
			results[i].String = cp

			for i := range set.Weights {
				w := set.Weights[i]
				if strings.HasPrefix(w.N, "b") || strings.HasPrefix(w.N, "x") {
					w.X = w.X[:cap(w.X)]
					w.States = make([][]float32, StateTotal)
					for ii := range w.States {
						w.States[ii] = make([]float32, len(w.X))
					}
					continue
				}
				factor := math.Sqrt(2.0 / float64(w.S[0]))
				for range cap(w.X) {
					w.X = append(w.X, float32(rng.NormFloat64()*factor))
				}
				w.States = make([][]float32, StateTotal)
				for ii := range w.States {
					w.States[ii] = make([]float32, len(w.X))
				}
			}

			dropout := map[string]interface{}{
				"rng": rng,
			}
			sum := tf32.Add(others.Get("x"), set.Get("y"))
			l1 := tf32.T(tf32.Mul(tf32.Dropout(tf32.Mul(sum, sum), dropout), tf32.T(sum)))
			loss := tf32.Avg(tf32.Quadratic(l1, set.Get("y")))

			for iteration := range 256 {
				pow := func(x float64) float64 {
					y := math.Pow(x, float64(iteration+1))
					if math.IsNaN(y) || math.IsInf(y, 0) {
						return 0
					}
					return y
				}

				l := float32(0.0)
				others.Zero()
				set.Zero()
				l = tf32.Gradient(loss).X[0]
				if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
					fmt.Println(iteration, l)
					panic("isnan or isinf")
				}
				//fmt.Println(iteration, l)

				norm := 0.0
				for _, p := range set.Weights {
					for _, d := range p.D {
						norm += float64(d * d)
					}
				}
				norm = math.Sqrt(norm)
				b1, b2 := pow(B1), pow(B2)
				scaling := 1.0
				if norm > 1 {
					scaling = 1 / norm
				}
				for _, w := range set.Weights {
					for ii, d := range w.D {
						g := d * float32(scaling)
						m := B1*w.States[StateM][ii] + (1-B1)*g
						v := B2*w.States[StateV][ii] + (1-B2)*g*g
						w.States[StateM][ii] = m
						w.States[StateV][ii] = v
						mhat := m / float32(1-b1)
						vhat := v / float32(1-b2)
						if vhat < 0 {
							vhat = 0
						}
						w.X[ii] -= Eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
					}
				}
			}

			y := set.ByName["y"]
			for ii := range length {
				yy := y.X[ii*256 : (ii+1)*256]
				softmax(yy)
				entropy := 0.0
				for _, value := range yy {
					entropy += float64(value) * math.Log2(float64(value))
				}
				results[i].Entropy += -entropy
			}
			fmt.Println("string", i, results[i].Entropy)
			done <- true
		}
		index, flight, cpus := 0, 0, runtime.NumCPU()
		for index < len(results) && flight < cpus {
			go process(rng.Int63(), index, str)
			index++
			flight++
		}
		for index < len(results) {
			<-done
			flight--

			go process(rng.Int63(), index, str)
			index++
			flight++
		}
		for range flight {
			<-done
		}

		sort.Slice(results, func(i, j int) bool {
			return results[i].Entropy > results[j].Entropy
		})
		/*for i := range results {
			fmt.Println(results[i].Entropy, results[i].String)
		}*/
		return results[0].String
	}
	for range 3 {
		str = process(str)
	}
	fmt.Println(string(str))
}

// GAMode genetic algorithm mode
func GAMode() {
	rng := rand.New(rand.NewSource(1))

	const (
		size       = 256
		step       = 128
		model      = 8
		Eta        = 1.0e-2
		population = 128
	)

	const (
		// S is the scaling factor for the softmax
		S = 1.0 - 1e-300
	)

	softmax := func(values []float32) {
		max := float32(0.0)
		/*for i, v := range values {
			if v < 0 {
				v = -v
			}
			values[i] = v
		}*/
		for _, v := range values {
			if v > max {
				max = v
			}
		}
		s := max * S
		sum := float32(0.0)
		for j, value := range values {
			values[j] = float32(math.Exp(float64(value - s)))
			sum += values[j]
		}
		for j, value := range values {
			values[j] = value / sum
		}
	}

	type File struct {
		Name  string
		Data  []byte
		Model order4.Model
	}

	files := []File{
		{Name: "pg74.txt.bz2"},
		{Name: "10.txt.utf-8.bz2"},
		{Name: "76.txt.utf-8.bz2"},
		{Name: "84.txt.utf-8.bz2"},
		{Name: "100.txt.utf-8.bz2"},
		{Name: "1837.txt.utf-8.bz2"},
		{Name: "2701.txt.utf-8.bz2"},
		{Name: "3176.txt.utf-8.bz2"},
		{Name: "all"},
	}
	files[len(files)-1].Model.Init()

	load := func(book *File, all *File) {
		path := fmt.Sprintf("books/%s", book.Name)
		file, err := Text.Open(path)
		if err != nil {
			panic(err)
		}
		defer file.Close()
		breader := bzip2.NewReader(file)
		data, err := io.ReadAll(breader)
		if err != nil {
			panic(err)
		}

		mark := order4.Markov{}
		book.Model.Init()
		for _, value := range data {
			for ii := range mark {
				vector := book.Model[ii][mark[ii]]
				if vector == nil {
					vector = make([]uint32, size)
				}
				vector[value]++
				book.Model[ii][mark[ii]] = vector

				{
					vector := all.Model[ii][mark[ii]]
					if vector == nil {
						vector = make([]uint32, size)
					}
					vector[value]++
					all.Model[ii][mark[ii]] = vector
				}

				state := value
				for iii, value := range mark[ii][:ii+1] {
					mark[ii][iii], state = state, value
				}
			}
		}
		book.Data = data
	}
	for i := range files[:len(files)-1] {
		load(&files[i], &files[len(files)-1])
		fmt.Println(files[i].Name)
		for ii := range files[i].Model {
			fmt.Println(len(files[i].Model[ii]))
		}
	}

	str := []byte("What is the meaning of life?")
	process := func(str []byte, model *order4.Model) ([]byte, *order4.Model) {
		type String struct {
			String  []byte
			Entropy float64
		}
		results := make([]String, population)
		done := make(chan bool, 8)
		process := func(seed int64, i int, str []byte) {
			rng := rand.New(rand.NewSource(seed))
			cp := make([]byte, len(str))
			copy(cp, str)
			length := len(cp) + step
			others := tf32.NewSet()
			others.Add("x", size, length)
			x := others.ByName["x"]

			set := tf32.NewSet()
			set.Add("y", size, length)

			var markov order4.Markov
			for _, value := range cp {
				markov.Iterate(value)
				distribution := model.Lookup(&markov)
				if distribution == nil {
					for range size {
						x.X = append(x.X, 0)
					}
					continue
				}
				for _, value := range distribution {
					x.X = append(x.X, value)
				}
			}
			for range step {
				distribution := model.Lookup(&markov)
				sum, selected := float32(0.0), rng.Float32()
				found := false
				for key, value := range distribution {
					sum += value
					if selected < sum {
						cp = append(cp, byte(key))
						markov.Iterate(byte(key))
						distribution := model.Lookup(&markov)
						for _, value := range distribution {
							x.X = append(x.X, value)
						}
						found = true
						break
					}
				}
				if !found {
					for range size {
						x.X = append(x.X, 0)
					}
				}
			}
			results[i].String = cp

			for i := range set.Weights {
				w := set.Weights[i]
				if strings.HasPrefix(w.N, "b") || strings.HasPrefix(w.N, "x") {
					w.X = w.X[:cap(w.X)]
					w.States = make([][]float32, StateTotal)
					for ii := range w.States {
						w.States[ii] = make([]float32, len(w.X))
					}
					continue
				}
				factor := math.Sqrt(2.0 / float64(w.S[0]))
				for range cap(w.X) {
					w.X = append(w.X, float32(rng.NormFloat64()*factor))
				}
				w.States = make([][]float32, StateTotal)
				for ii := range w.States {
					w.States[ii] = make([]float32, len(w.X))
				}
			}

			dropout := map[string]interface{}{
				"rng": rng,
			}
			sum := tf32.Add(others.Get("x"), set.Get("y"))
			l1 := tf32.T(tf32.Mul(tf32.Dropout(tf32.Mul(sum, sum), dropout), tf32.T(sum)))
			loss := tf32.Avg(tf32.Quadratic(l1, set.Get("y")))

			for iteration := range 128 {
				pow := func(x float64) float64 {
					y := math.Pow(x, float64(iteration+1))
					if math.IsNaN(y) || math.IsInf(y, 0) {
						return 0
					}
					return y
				}

				l := float32(0.0)
				others.Zero()
				set.Zero()
				l = tf32.Gradient(loss).X[0]
				if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
					fmt.Println(iteration, l)
					panic("isnan or isinf")
				}
				//fmt.Println(iteration, l)

				norm := 0.0
				for _, p := range set.Weights {
					for _, d := range p.D {
						norm += float64(d * d)
					}
				}
				norm = math.Sqrt(norm)
				b1, b2 := pow(B1), pow(B2)
				scaling := 1.0
				if norm > 1 {
					scaling = 1 / norm
				}
				for _, w := range set.Weights {
					for ii, d := range w.D {
						g := d * float32(scaling)
						m := B1*w.States[StateM][ii] + (1-B1)*g
						v := B2*w.States[StateV][ii] + (1-B2)*g*g
						w.States[StateM][ii] = m
						w.States[StateV][ii] = v
						mhat := m / float32(1-b1)
						vhat := v / float32(1-b2)
						if vhat < 0 {
							vhat = 0
						}
						w.X[ii] -= Eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
					}
				}
			}

			y := set.ByName["y"]
			avg := make([]float32, size)
			for ii := range length {
				yy := y.X[ii*size : (ii+1)*size]
				for iii, v := range yy {
					avg[iii] += v
				}
			}
			for ii := range avg {
				avg[ii] /= float32(length)
			}
			vari := make([]float32, size)
			for ii := range length {
				yy := y.X[ii*size : (ii+1)*size]
				for iii, v := range yy {
					diff := avg[iii] - v
					vari[iii] += diff * diff
				}
			}
			for ii, v := range vari {
				vari[ii] = float32(math.Sqrt(float64(v / float32(length))))
			}
			softmax(vari)
			entropy := 0.0
			for _, value := range vari {
				entropy += float64(value) * math.Log2(float64(value))
			}
			results[i].Entropy = -entropy
			fmt.Println("string", i, results[i].Entropy)
			done <- true
		}
		index, flight, cpus := 0, 0, runtime.NumCPU()
		for index < len(results) && flight < cpus {
			go process(rng.Int63(), index, str)
			index++
			flight++
		}
		for index < len(results) {
			<-done
			flight--

			go process(rng.Int63(), index, str)
			index++
			flight++
		}
		for range flight {
			<-done
		}

		sort.Slice(results, func(i, j int) bool {
			return results[i].Entropy > results[j].Entropy
		})
		/*for i := range results {
			fmt.Println(results[i].Entropy, results[i].String)
		}*/
		m := order4.Model{}
		m.Init()
		for i := range results[:len(results)/2] {
			markov := order4.Markov{}
			for _, value := range results[i].String[len(str):] {
				for ii := range markov {
					vector := m[ii][markov[ii]]
					if vector == nil {
						vector = make([]uint32, size)
					}
					vector[value]++
					m[ii][markov[ii]] = vector
					state := value
					for iii, value := range markov[ii][:ii+1] {
						markov[ii][iii], state = state, value
					}
				}
			}
		}
		return results[0].String, &m
	}
	var s []byte
	m := &files[model].Model
	for range 3 {
		s, m = process(str, m)
	}
	fmt.Println(string(s))
}

func main() {
	flag.Parse()

	if *FlagCluster {
		ClusterMode()
		return
	}

	if *FlagLM {
		LMMode()
		return
	}

	if *FlagGA {
		GAMode()
		return
	}
	rng := rand.New(rand.NewSource(1))

	const (
		size       = 2
		step       = 1024
		model      = 8
		Eta        = 1.0e-2
		population = 256
	)

	const (
		// S is the scaling factor for the softmax
		S = 1.0 - 1e-300
	)

	softmax := func(values []float32) {
		max := float32(0.0)
		/*for i, v := range values {
			if v < 0 {
				v = -v
			}
			values[i] = v
		}*/
		for _, v := range values {
			if v > max {
				max = v
			}
		}
		s := max * S
		sum := float32(0.0)
		for j, value := range values {
			values[j] = float32(math.Exp(float64(value - s)))
			sum += values[j]
		}
		for j, value := range values {
			values[j] = value / sum
		}
	}

	type File struct {
		Name  string
		Data  []byte
		Model order.Model
	}

	files := []File{
		{Name: "pg74.txt.bz2"},
		{Name: "10.txt.utf-8.bz2"},
		{Name: "76.txt.utf-8.bz2"},
		{Name: "84.txt.utf-8.bz2"},
		{Name: "100.txt.utf-8.bz2"},
		{Name: "1837.txt.utf-8.bz2"},
		{Name: "2701.txt.utf-8.bz2"},
		{Name: "3176.txt.utf-8.bz2"},
		{Name: "all"},
	}
	files[len(files)-1].Model.Init()

	load := func(book *File, all *File) {
		path := fmt.Sprintf("books/%s", book.Name)
		file, err := Text.Open(path)
		if err != nil {
			panic(err)
		}
		defer file.Close()
		breader := bzip2.NewReader(file)
		d, err := io.ReadAll(breader)
		if err != nil {
			panic(err)
		}
		data := make([]byte, 0, len(d)*8)
		for _, value := range d {
			for i := range 8 {
				data = append(data, byte((value>>(7-i))&1))
			}
		}

		mark := order.Markov{}
		book.Model.Init()
		for _, value := range data {
			for ii := range mark {
				/*vector := book.Model[ii][mark[ii]]
				if vector == nil {
					vector = make([]uint32, size)
				}
				vector[value]++
				book.Model[ii][mark[ii]] = vector*/

				{
					vector := all.Model[ii][mark[ii]]
					if vector == nil {
						vector = make([]uint32, size)
					}
					vector[value]++
					all.Model[ii][mark[ii]] = vector
				}

				state := value
				for iii, value := range mark[ii][:ii+1] {
					mark[ii][iii], state = state, value
				}
			}
		}
		book.Data = data
	}
	for i := range files[:len(files)-1] {
		load(&files[i], &files[len(files)-1])
		fmt.Println(files[i].Name)
		for ii := range files[i].Model {
			fmt.Println(len(files[i].Model[ii]))
		}
	}

	str := []byte("What is the meaning of life?")
	process := func(str []byte, model *order.Model) ([]byte, *order.Model) {
		type String struct {
			String  []byte
			Entropy float64
		}
		results := make([]String, population)
		done := make(chan bool, 8)
		process := func(seed int64, i int, str []byte) {
			rng := rand.New(rand.NewSource(seed))
			cp := make([]byte, 0, len(str)*8)
			for _, value := range str {
				for i := range 8 {
					cp = append(cp, byte((value>>(7-i))&1))
				}
			}
			length := len(cp) + step
			others := tf32.NewSet()
			others.Add("x", size, length)
			x := others.ByName["x"]

			set := tf32.NewSet()
			set.Add("y", size, length)

			var markov order.Markov
			for _, value := range cp {
				markov.Iterate(value)
				distribution := model.Lookup(&markov)
				if distribution == nil {
					for range size {
						x.X = append(x.X, 0)
					}
					continue
				}
				for _, value := range distribution {
					x.X = append(x.X, value)
				}
			}
			for range step {
				distribution := model.Lookup(&markov)
				sum, selected := float32(0.0), rng.Float32()
				found := false
				for key, value := range distribution {
					sum += value
					if selected < sum {
						cp = append(cp, byte(key))
						markov.Iterate(byte(key))
						distribution := model.Lookup(&markov)
						for _, value := range distribution {
							x.X = append(x.X, value)
						}
						found = true
						break
					}
				}
				if !found {
					for range size {
						x.X = append(x.X, 0)
					}
				}
			}
			results[i].String = cp

			for i := range set.Weights {
				w := set.Weights[i]
				if strings.HasPrefix(w.N, "b") || strings.HasPrefix(w.N, "x") {
					w.X = w.X[:cap(w.X)]
					w.States = make([][]float32, StateTotal)
					for ii := range w.States {
						w.States[ii] = make([]float32, len(w.X))
					}
					continue
				}
				factor := math.Sqrt(2.0 / float64(w.S[0]))
				for range cap(w.X) {
					w.X = append(w.X, float32(rng.NormFloat64()*factor))
				}
				w.States = make([][]float32, StateTotal)
				for ii := range w.States {
					w.States[ii] = make([]float32, len(w.X))
				}
			}

			dropout := map[string]interface{}{
				"rng": rng,
			}
			sum := tf32.Add(others.Get("x"), set.Get("y"))
			l1 := tf32.T(tf32.Mul(tf32.Dropout(tf32.Mul(sum, sum), dropout), tf32.T(sum)))
			loss := tf32.Avg(tf32.Quadratic(l1, set.Get("y")))

			for iteration := range 8 {
				pow := func(x float64) float64 {
					y := math.Pow(x, float64(iteration+1))
					if math.IsNaN(y) || math.IsInf(y, 0) {
						return 0
					}
					return y
				}

				l := float32(0.0)
				others.Zero()
				set.Zero()
				l = tf32.Gradient(loss).X[0]
				if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
					fmt.Println(iteration, l)
					panic("isnan or isinf")
				}
				//fmt.Println(iteration, l)

				norm := 0.0
				for _, p := range set.Weights {
					for _, d := range p.D {
						norm += float64(d * d)
					}
				}
				norm = math.Sqrt(norm)
				b1, b2 := pow(B1), pow(B2)
				scaling := 1.0
				if norm > 1 {
					scaling = 1 / norm
				}
				for _, w := range set.Weights {
					for ii, d := range w.D {
						g := d * float32(scaling)
						m := B1*w.States[StateM][ii] + (1-B1)*g
						v := B2*w.States[StateV][ii] + (1-B2)*g*g
						w.States[StateM][ii] = m
						w.States[StateV][ii] = v
						mhat := m / float32(1-b1)
						vhat := v / float32(1-b2)
						if vhat < 0 {
							vhat = 0
						}
						w.X[ii] -= Eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
					}
				}
			}

			y := set.ByName["y"]
			avg := make([]float32, size)
			for ii := range length {
				yy := y.X[ii*size : (ii+1)*size]
				for iii, v := range yy {
					avg[iii] += v
				}
			}
			for ii := range avg {
				avg[ii] /= float32(length)
			}
			vari := make([]float32, size)
			for ii := range length {
				yy := y.X[ii*size : (ii+1)*size]
				for iii, v := range yy {
					diff := avg[iii] - v
					vari[iii] += diff * diff
				}
			}
			for ii, v := range vari {
				vari[ii] = float32(math.Sqrt(float64(v / float32(length))))
			}
			softmax(vari)
			entropy := 0.0
			for _, value := range vari {
				entropy += float64(value) * math.Log2(float64(value))
			}
			results[i].Entropy = -entropy
			fmt.Println("string", i, results[i].Entropy)
			done <- true
		}
		index, flight, cpus := 0, 0, runtime.NumCPU()
		for index < len(results) && flight < cpus {
			go process(rng.Int63(), index, str)
			index++
			flight++
		}
		for index < len(results) {
			<-done
			flight--

			go process(rng.Int63(), index, str)
			index++
			flight++
		}
		for range flight {
			<-done
		}

		sort.Slice(results, func(i, j int) bool {
			return results[i].Entropy > results[j].Entropy
		})
		/*for i := range results {
			fmt.Println(results[i].Entropy, results[i].String)
		}*/
		m := order.Model{}
		m.Init()
		for i := range results[:len(results)/2] {
			markov := order.Markov{}
			for _, value := range results[i].String {
				for ii := range markov {
					vector := m[ii][markov[ii]]
					if vector == nil {
						vector = make([]uint32, size)
					}
					vector[value]++
					m[ii][markov[ii]] = vector
					state := value
					for iii, value := range markov[ii][:ii+1] {
						markov[ii][iii], state = state, value
					}
				}
			}
		}
		return results[0].String, &m
	}
	var s []byte
	m := &files[model].Model
	for range 3 {
		s, m = process(str, m)
	}
	data := make([]byte, 0, len(s)/8)
	var acc byte
	for i, value := range s {
		if i%8 == 0 && i != 0 {
			data = append(data, acc)
			acc = 0
		}
		acc <<= 1
		acc |= value
	}
	fmt.Println(string(data))
}
