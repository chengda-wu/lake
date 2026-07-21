// lake-router:P3 OpenAI 兼容 HTTP 入口。
//
//	go run ./router/cmd/router
//
// 环境变量:LAKE_HTTP_ADDR / LAKE_WORKER_ADDR / LAKE_AGENT_ADDR
package main

import (
	"log"
	"os"

	"github.com/chengda-wu/lake/go/router"
)

func main() {
	cfg := router.Config{
		HTTPAddr:   env("LAKE_HTTP_ADDR", ":8080"),
		WorkerAddr: env("LAKE_WORKER_ADDR", "127.0.0.1:50053"),
		AgentAddr:  env("LAKE_AGENT_ADDR", "127.0.0.1:50054"),
	}
	s, err := router.New(cfg)
	if err != nil {
		log.Fatal(err)
	}
	log.Fatal(s.ListenAndServe())
}

func env(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
