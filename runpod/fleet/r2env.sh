# Source in ssh sessions: the container's R2 settings live in PID 1's environment. Prints nothing.
while IFS= read -r -d '' kv; do
  case "$kv" in R2_ENDPOINT_URL=*|R2_BUCKET=*|AWS_ACCESS_KEY_ID=*|AWS_SECRET_ACCESS_KEY=*) export "$kv";; esac
done < /proc/1/environ
