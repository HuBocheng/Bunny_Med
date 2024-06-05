export HF_ENDPOINT=https://hf-mirror.com

access_token=$(<access_token.txt)


# huggingface-cli download \
# --resume-download microsoft/phi-2 \
# --local-dir ./weight/phi-2 \
# --local-dir-use-symlinks False \
# --token $access_token

# huggingface-cli download \
# --resume-download google/siglip-so400m-patch14-384 \
# --local-dir ./weight/siglip \
# --local-dir-use-symlinks False \
# --token $access_token


# huggingface-cli download \
# --resume-download BAAI/Bunny-v1_0-3B \
# --local-dir ./weight/Bunny-phi-siglip \
# --local-dir-use-symlinks False \
# --token $access_token


# huggingface-cli download \
# --resume-download BAAI/bunny-pretrain-phi-2-siglip \
# --local-dir ./weight/bunny-pretrain-phi-2-siglip \
# --local-dir-use-symlinks False \
# --token $access_token

huggingface-cli download \
--repo-type dataset \
--resume-download BoyaWu10/Bunny-v1_0-data \
--local-dir-use-symlinks False \
--token $access_token
