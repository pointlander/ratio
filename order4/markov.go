// Copyright 2025 The Ratio Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package order4

const (
	Order = 4
)

type markov [Order]byte
type Markov [Order]markov
type Model [Order]map[markov][]uint32

func (m *Model) Init() {
	for i := range m {
		m[i] = make(map[markov][]uint32)
	}
}

// Lookup looks a vector up
func (m *Model) Lookup(markov *Markov) []float32 {
	for i := range markov {
		i = Order - 1 - i
		vector := m[i][markov[i]]
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
func (m *Markov) Iterate(state byte) {
	for i := range m {
		state := state
		for ii, value := range m[i][:i+1] {
			m[i][ii], state = state, value
		}
	}
}
