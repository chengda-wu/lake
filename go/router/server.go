package router

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"

	lakepb "github.com/chengda-wu/lake/go/pb"
)

// Config P3 Router 配置。
type Config struct {
	HTTPAddr   string // 默认 :8080
	WorkerAddr string // WorkerService,默认 127.0.0.1:50053
}

// Server OpenAI 兼容 HTTP → gRPC Worker.Generate。
// P3 入口即本服务(边2);不经 Bifrost。
// LookupPrefix 在 worker 侧完成(直连 ControlPlane),Router 不持 CP client。
type Server struct {
	cfg    Config
	worker lakepb.WorkerServiceClient
}

func New(cfg Config) (*Server, error) {
	if cfg.HTTPAddr == "" {
		cfg.HTTPAddr = ":8080"
	}
	if cfg.WorkerAddr == "" {
		cfg.WorkerAddr = "127.0.0.1:50053"
	}
	wconn, err := grpc.NewClient(cfg.WorkerAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("dial worker: %w", err)
	}
	return &Server{
		cfg:    cfg,
		worker: lakepb.NewWorkerServiceClient(wconn),
	}, nil
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})
	mux.HandleFunc("/v1/chat/completions", s.handleChatCompletions)
	return mux
}

func (s *Server) ListenAndServe() error {
	log.Printf("lake-router OpenAI HTTP on %s → worker %s", s.cfg.HTTPAddr, s.cfg.WorkerAddr)
	return http.ListenAndServe(s.cfg.HTTPAddr, s.Handler())
}

type chatRequest struct {
	Model    string `json:"model"`
	Messages []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"messages"`
	MaxTokens int `json:"max_tokens"`
}

type chatResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	// lake 扩展:前缀复用统计(冒烟用)
	Lake struct {
		ReusedBlocks  uint32 `json:"reused_blocks"`
		PrefillBlocks uint32 `json:"prefill_blocks"`
		Mode          string `json:"mode"`
	} `json:"lake"`
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	var req chatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	promptText := ""
	for _, m := range req.Messages {
		if m.Role == "user" || m.Role == "system" {
			if promptText != "" {
				promptText += "\n"
			}
			promptText += m.Content
		}
	}
	tokens := tokenizeMock(promptText)
	maxNew := req.MaxTokens
	if maxNew <= 0 {
		maxNew = 4
	}
	model := req.Model
	if model == "" {
		model = "mock-llm"
	}
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	rid := uuid.NewString()
	gen, err := s.worker.Generate(ctx, &lakepb.GenerateRequest{
		RequestId:       rid,
		ModelId:         model,
		PromptTokens:    tokens,
		MaxNewTokens:    uint32(maxNew),
		RequesterNodeId: "worker-0",
	})
	if err != nil {
		code, msg := mapGRPCError(err)
		http.Error(w, msg, code)
		return
	}

	content := detokenizeMock(gen.OutputTokens)
	resp := chatResponse{
		ID:      "chatcmpl-" + rid[:8],
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
	}
	resp.Choices = make([]struct {
		Index   int `json:"index"`
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	}, 1)
	resp.Choices[0].Index = 0
	resp.Choices[0].Message.Role = "assistant"
	resp.Choices[0].Message.Content = content
	resp.Choices[0].FinishReason = "stop"
	resp.Lake.ReusedBlocks = gen.ReusedBlocks
	resp.Lake.PrefillBlocks = gen.PrefillBlocks
	resp.Lake.Mode = gen.Mode

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func mapGRPCError(err error) (httpStatus int, msg string) {
	st, ok := status.FromError(err)
	if !ok {
		return http.StatusBadGateway, "Generate: " + err.Error()
	}
	msg = fmt.Sprintf("Generate: %s", st.Message())
	switch st.Code() {
	case codes.InvalidArgument:
		return http.StatusBadRequest, msg
	case codes.Unavailable, codes.DeadlineExceeded:
		return http.StatusServiceUnavailable, msg
	default:
		return http.StatusBadGateway, msg
	}
}

// tokenizeMock:P3 跳过真实 tokenizer,按 rune 映射到稳定 uint32。
func tokenizeMock(s string) []uint32 {
	s = strings.TrimSpace(s)
	if s == "" {
		return []uint32{1, 2, 3, 4, 5, 6, 7, 8} // 至少一块
	}
	out := make([]uint32, 0, len(s))
	for _, r := range s {
		out = append(out, uint32(r)%10000+1)
	}
	// 补齐到 block 边界(8),稳定复用
	for len(out)%8 != 0 {
		out = append(out, 42)
	}
	return out
}

func detokenizeMock(tokens []uint32) string {
	parts := make([]string, len(tokens))
	for i, t := range tokens {
		parts[i] = fmt.Sprintf("%d", t)
	}
	return "mock:" + strings.Join(parts, ",")
}
