#!/usr/bin/env bash
# 首先停止所有服务, kill 所有存在的僵尸进程 防止端口占用或者 其他冲突
ps -ef | grep fastchat.serve | awk '{print $2}' |xargs -I{} kill -9 {}

ps -ef | grep gpt_server | awk '{print $2}' |xargs -I{} kill -9 {}

python main.py