# Sourced by pixi on every `pixi run` and `pixi shell`.
if [ -f "$PIXI_PROJECT_ROOT/config/demo.env" ]; then
  set -a
  . "$PIXI_PROJECT_ROOT/config/demo.env"
  set +a
fi
export CANDETECT_METAFLOW_ENV="${CANDETECT_METAFLOW_ENV:-fast-bakery}"
