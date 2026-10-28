package axengine

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
)

// UnmarshalJSON accepts both embedding encodings the server returns: a JSON
// number array (encoding_format "float") and a base64 string of
// little-endian float32 values (encoding_format "base64"). Either way the
// vector lands in Embedding.
func (o *OpenAiEmbeddingObject) UnmarshalJSON(data []byte) error {
	var raw struct {
		Object    string          `json:"object"`
		Embedding json.RawMessage `json:"embedding"`
		Index     int             `json:"index"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	o.Object = raw.Object
	o.Index = raw.Index
	o.Embedding = nil
	if len(raw.Embedding) == 0 || string(raw.Embedding) == "null" {
		return nil
	}
	if raw.Embedding[0] != '"' {
		return json.Unmarshal(raw.Embedding, &o.Embedding)
	}
	var encoded string
	if err := json.Unmarshal(raw.Embedding, &encoded); err != nil {
		return err
	}
	bytes, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return fmt.Errorf("axengine: invalid base64 embedding: %w", err)
	}
	if len(bytes)%4 != 0 {
		return fmt.Errorf("axengine: base64 embedding is %d bytes, not a float32 multiple", len(bytes))
	}
	o.Embedding = make([]float64, len(bytes)/4)
	for i := range o.Embedding {
		o.Embedding[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(bytes[i*4:])))
	}
	return nil
}
