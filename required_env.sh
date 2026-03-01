required_env_vars=(
  "PYPI_USERNAME"
  "PYPI_PASSWORD"
  "CODEBERG_USERNAME"
  "CODEBERG_PASSWORD"
  "ITCH_TOKEN"
  "LISIEN_VERSION"
);
for required_env in "${required_env_vars[@]}"; do
  if [ -z "${!required_env}" ]; then
    echo "Required environment variable not set: ${required_env}";
    exit 1;
  fi;
done;