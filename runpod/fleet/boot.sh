#!/bin/bash
# boot.sh POD_ID INDEX TOTAL — upload setup + code, then setup and run rotation INDEX/TOTAL. Logs on the pod.
set -u
cd $(dirname "$0")
read IP PORT < <(python3 podssh.py "$1") || { echo "$1 no ssh"; exit 1; }
S="ssh -i $(dirname "$0")/pod_key -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$(dirname "$0")/known_hosts -o ConnectTimeout=20 -o ServerAliveInterval=30 -p $PORT root@$IP"
echo "$1 $IP $PORT" > $(dirname "$0")/ssh-$1.txt
for i in 1 2 3 4 5 6; do $S true 2>/dev/null && break; sleep 10; done
for f in r2env.sh setup.sh code.tgz run_sapiens.sh; do $S "cat > /root/$f" < $(dirname "$0")/$f; done
$S "nohup bash -c 'bash /root/setup.sh > /root/setup.log 2>&1 && bash /root/run_sapiens.sh $2 $3 > /root/sapiens.log 2>&1' > /dev/null 2>&1 &"
echo "$1 booted rotation $2/$3"
