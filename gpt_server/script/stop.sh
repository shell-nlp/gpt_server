#!/usr/bin/env bash

# ps -ef | grep fastchat.serve | awk '{print $2}' |xargs -I{} kill -9 {}

ps -ef | grep gpt_server | awk '{print $2}' |xargs -I{} kill -9 {}