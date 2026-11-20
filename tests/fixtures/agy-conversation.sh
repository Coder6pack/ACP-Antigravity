#!/bin/sh
if [ "$1" = models ]; then
    printf 'smoke-model\n'
    exit 0
fi
conversation=none
log_path=
while [ "$#" -gt 0 ]; do
    case "$1" in
        --conversation) conversation=$2; shift ;;
        --log-file) log_path=$2; shift ;;
    esac
    shift
done
IFS= read -r prompt
case "$prompt" in
    *replace-conversation*|*follow-replacement*) next=$ACP_TEST_REPLACEMENT_ID ;;
    *) next=$ACP_TEST_FIRST_ID ;;
esac
printf 'Created conversation %s\n' "$next" > "$log_path"
printf '{"event":"result","conversation_id":"%s","result":{"status":"SUCCESS","response":"received:%s"}}\n' "$next" "$conversation"
