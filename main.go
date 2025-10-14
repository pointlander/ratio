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
	"sort"
	"strconv"
	"strings"

	"github.com/pointlander/ratio/kmeans"

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

const (
	order = 4
)

type Markov [order]byte
type Model [order]map[Markov][]uint32

// Lookup looks a vector up
func Lookup(markov *[order]Markov, model *Model) []float32 {
	for i := range markov {
		i = order - 1 - i
		vector := model[i][markov[i]]
		if vector != nil {
			sum := float32(0.0)
			for _, value := range vector {
				sum += float32(value)
			}
			result := make([]float32, len(vector))
			for ii, value := range vector {
				result[ii] = float32(value) / sum
			}
			return result
		}
	}
	return nil
}

// Iterate iterates a markov model
func Iterate(markov *[order]Markov, state byte) {
	for i := range markov {
		state := state
		for ii, value := range markov[i][:i+1] {
			markov[i][ii], state = state, value
		}
	}
}

var (
	// FlagCluster cluster mode
	FlagCluster = flag.Bool("c", false, "cluster mode")
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

func main() {
	flag.Parse()

	if *FlagCluster {
		ClusterMode()
		return
	}

	rng := rand.New(rand.NewSource(1))

	const (
		size = 256
	)

	type File struct {
		Name  string
		Data  []byte
		Model Model
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
	}

	load := func(book *File) {
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

		markov := [order]Markov{}
		for i := range book.Model {
			book.Model[i] = make(map[Markov][]uint32)
		}
		for _, value := range data {
			for ii := range markov {
				vector := book.Model[ii][markov[ii]]
				if vector == nil {
					vector = make([]uint32, size)
				}
				vector[value]++
				book.Model[ii][markov[ii]] = vector

				state := value
				for iii, value := range markov[ii][:ii+1] {
					markov[ii][iii], state = state, value
				}
			}
		}
		book.Data = data
	}
	for i := range files {
		load(&files[i])
		fmt.Println(files[i].Name)
		for ii := range files[i].Model {
			fmt.Println(len(files[i].Model[ii]))
		}
	}

	str := []byte("What is the meaning of life?")
	length := len(str) //+ 128
	set := tf64.NewSet()
	set.Add("y", 256, length)
	set.Add("x", 256, length)
	x := set.ByName["x"]

	var markov [order]Markov
	for _, value := range str {
		Iterate(&markov, value)
		distribution := Lookup(&markov, &files[1].Model)
		for _, value := range distribution {
			x.X = append(x.X, float64(value))
		}
	}
	/*for range 128 {
		distribution := Lookup(&markov, &files[1].Model)
		sum, selected := float32(0.0), rng.Float32()
		for key, value := range distribution {
			sum += value
			if selected < sum {
				str = append(str, byte(key))
				Iterate(&markov, byte(key))
				distribution := Lookup(&markov, &files[1].Model)
				for _, value := range distribution {
					x.X = append(x.X, float64(value))
				}
				break
			}
		}
	}*/

	for i := range set.Weights {
		w := set.Weights[i]
		if strings.HasPrefix(w.N, "b") || strings.HasPrefix(w.N, "x") {
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

	dropout := map[string]interface{}{
		"rng": rng,
	}

	for iteration := range 128 {
		pow := func(x float64) float64 {
			y := math.Pow(x, float64(iteration+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return y
		}
		xx := tf64.Softmax(set.Get("x"))
		sum := tf64.Add(xx, set.Get("y"))
		l1 := tf64.T(tf64.Mul(tf64.Dropout(tf64.Mul(sum, sum), dropout), tf64.T(sum)))
		loss := tf64.Add(tf64.Avg(tf64.Quadratic(l1, set.Get("y"))), tf64.Avg(tf64.Quadratic(l1, xx)))

		l := 0.0
		set.Zero()
		l = tf64.Gradient(loss).X[0]
		if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
			fmt.Println(iteration, l)
			return
		}
		fmt.Println(iteration, l)

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
		const Eta = 1.0e-3
		for _, w := range set.Weights {
			/*if strings.HasPrefix(w.N, "x") {
				for ii, d := range w.D[len(str):] {
					ii += len(str)
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
				continue
			}*/
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

	/*cs := func(a []float32, b []float64) float64 {
		ab, aa, bb := 0.0, 0.0, 0.0
		for key, value := range a {
			ab += float64(value) * b[key]
		}
		for _, value := range a {
			aa += float64(value) * float64(value)
		}
		if aa <= 0 {
			return 0
		}
		for _, value := range b {
			bb += value * value
		}
		if bb <= 0 {
			return 0
		}
		return ab / (math.Sqrt(aa) * math.Sqrt(bb))
	}*/
	const (
		// S is the scaling factor for the softmax
		S = 1.0 - 1e-300
	)

	softmax := func(values []float64) {
		max := 0.0
		for _, v := range values {
			if v > max {
				max = v
			}
		}
		s := max * S
		sum := 0.0
		for j, value := range values {
			values[j] = math.Exp(value - s)
			sum += values[j]
		}
		for j, value := range values {
			values[j] = value / sum
		}
	}

	stri := []byte{}
	for i := range length {
		xx := x.X[i*256 : (i+1)*256]
		softmax(xx)
		total, selected := 0.0, rng.Float64()
		for key, value := range xx {
			total += value
			if selected < total {
				stri = append(stri, byte(key))
				break
			}
		}
	}

	fmt.Println(string(str))
	/*stri := []byte("What is the meaning of life?")
	{
		var markov [order]Markov
		for _, value := range stri {
			Iterate(&markov, value)
		}
		index := len(stri)
		for index < length {
			samples := make([]float64, 256)
			for s := range samples {
				m := markov
				Iterate(&m, byte(s))
				distribution := Lookup(&m, &files[1].Model)
				samples[s] = cs(distribution, x.X[index*256:(index+1)*256])
			}
			sum := 0.0
			for key, value := range samples {
				if value < 0 {
					value = -value
				}
				sum += value
				samples[key] = value
			}
			for i := range samples {
				samples[i] /= sum
			}
			total, selected := 0.0, rng.Float64()
			for key, value := range samples {
				total += value
				if selected < total {
					Iterate(&markov, byte(key))
					stri = append(stri, byte(key))
					break
				}
			}
			index++
		}
	}*/
	fmt.Println(string(stri))
}
