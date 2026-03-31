#!/bin/bash
set -e

# Write authorized keys from environment variable
if [ -n "$AUTHORIZED_KEYS" ]; then
    echo "$AUTHORIZED_KEYS" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi

# Generate host keys if missing
ssh-keygen -A 2>/dev/null || true

echo "Starting SSH server..."
exec /usr/sbin/sshd -D -e
