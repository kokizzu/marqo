#!/bin/bash
#source /opt/bash-utils/logger.sh
export PYTHONPATH="${PYTHONPATH}:/app/src/"
if [ -z "${MARQO_CUDA_PATH}" ]; then
    export CUDA_HOME=/usr/local/cuda
else
    export CUDA_HOME=${MARQO_CUDA_PATH}
fi
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}

trap "bash /app/scripts/shutdown.sh; exit" SIGTERM SIGINT

function wait_for_process () {
    local max_retries=30
    local n_restarts_before_sigkill=3
    local process_name="$1"
    local retries=0
    while ! [[ $(docker ps -a | grep CONTAINER) ]] >/dev/null && ((retries < max_retries)); do
        echo "Process $process_name is not running yet. Retrying in 1 seconds"
        echo "Retry $retries of a maximum of $max_retries retries"
        ((retries=retries+1))
        if ((retries >= n_restarts_before_sigkill)); then
            echo "sending SIGKILL to dockerd and restarting "
            ps axf | grep docker | grep -v grep | awk '{print "kill -9 " $1}' | sh; rm /var/run/docker.pid; dockerd &
        else
            dockerd &
        fi
        sleep 3
        if ((retries >= max_retries)); then
            return 1
        fi
    done
    return 0
}


VESPA_IS_INTERNAL=False
# Vespa local run
if ([ -n "$VESPA_QUERY_URL" ] || [ -n "$VESPA_DOCUMENT_URL" ] || [ -n "$VESPA_CONFIG_URL" ]) && \
   ([ -z "$VESPA_QUERY_URL" ] || [ -z "$VESPA_DOCUMENT_URL" ] || [ -z "$VESPA_CONFIG_URL" ]); then
  echo "Error: Partial external vector store configuration detected. \
Please provide all or none of the VESPA_QUERY_URL, VESPA_DOCUMENT_URL, VESPA_CONFIG_URL. \
See https://docs.marqo.ai/2.0.0/Guides/Advanced-Usage/configuration/ for more information"
  exit 1

elif [ -z "$VESPA_QUERY_URL" ] && [ -z "$VESPA_DOCUMENT_URL" ] && [ -z "$VESPA_CONFIG_URL" ]; then
  # Start local vespa
  echo "External vector store not configured. Using local vector store"
  chown -R vespa:vespa /opt/vespa/var/
  tmux new-session -d -s vespa "bash /usr/local/bin/start_vespa.sh"

  echo "Waiting for vector store to start"
  for i in {1..5}; do
    if [ $i -eq 1 ]; then
      suffix="second"
    else
      suffix="seconds"
    fi
    echo -ne "Waiting... $i $suffix\r"
    sleep 1
  done

  # Try to deploy the application and branch on the output
  END_POINT="http://localhost:19071/application/v2/tenant/default/application/default"
  MAX_RETRIES=10
  RETRY_COUNT=0

  while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    # Make the curl request and capture the output
    RESPONSE=$(curl -s -X GET "$END_POINT")

    # Check for the specific "not found" error response which indicates there is no application package deployed
    if echo "$RESPONSE" | grep -q '"error-code":"NOT_FOUND"'; then
      echo "Marqo did not find an existing vector store. Setting up vector store..."

      # Generate and deploy the application package
      python3 /app/scripts/vespa_local/vespa_local.py generate-and-deploy

      until curl -f -X GET http://localhost:8080 >/dev/null 2>&1; do
        echo "  Waiting for vector store to be available..."
        sleep 10
      done
      echo "  Vector store is available. Vector store setup complete"
      break

    # Check for the "generation" success response which indicates there is an existing application package deployed
    elif echo "$RESPONSE" | grep -q '"generation":'; then
      echo "Marqo found an existing vector store. Waiting for vector store to be available..."

      until curl -f -X GET http://localhost:8080 >/dev/null 2>&1; do
        echo "  Waiting for vector store to be available..."
        sleep 10
      done
      echo "  Vector store is available. Vector store setup complete"
      break
    fi
    ((RETRY_COUNT++))
    sleep 5
  done

  if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "Warning: Failed to configure local vector store. Marqo may not function correctly"
  fi

  export VESPA_QUERY_URL="http://localhost:8080"
  export VESPA_DOCUMENT_URL="http://localhost:8080"
  export VESPA_CONFIG_URL="http://localhost:19071"
  export ZOOKEEPER_HOSTS="localhost:2181"
  export VESPA_IS_INTERNAL=True

else
  echo "External vector store configured. Using external vector store"
fi

# Start up redis
if [ "$MARQO_ENABLE_THROTTLING" != "FALSE" ]; then
    echo "Starting Marqo throttling"
    redis-server /etc/redis/redis.conf &
    echo "Called Marqo throttling start command"

    start_time=$(($(date +%s%N)/1000000))
    while true; do
        redis-cli ping &> /dev/null
        if [ $? -eq 0 ]; then
            break
        fi

        current_time=$(($(date +%s%N)/1000000))
        elapsed_time=$(expr $current_time - $start_time)
        if [ $elapsed_time -ge 2000 ]; then
            # Expected start time should be < 30ms in reality.
            echo "Marqo throttling server failed to start within 2s. skipping"
            break
        fi
        sleep 0.1
        
    done
    echo "Marqo throttling is now running"

else
    echo "Throttling has been disabled. Skipping Marqo throttling start"
fi

# set the default value to info and convert to lower case
export MARQO_LOG_LEVEL=${MARQO_LOG_LEVEL:-info}
MARQO_LOG_LEVEL=`echo "$MARQO_LOG_LEVEL" | tr '[:upper:]' '[:lower:]'`

# Start the tensor search web app in the background
cd /app/src/marqo/tensor_search || exit
uvicorn api:app --host 0.0.0.0 --port 8882 --timeout-keep-alive 75 --log-level $MARQO_LOG_LEVEL &
export api_pid=$!
wait "$api_pid"


# Exit with status of process that exited first
exit $?
