#!/usr/bin/env bash
# Готовит контекст сборки: снимок исходников MagicBrain фиксированной ревизии.
#
# Берётся именно `git archive HEAD`, а не рабочее дерево: иначе в образ попадут
# чужие незакоммиченные правки, и записанный в прогонах code_sha будет врать.
# Если дерево грязное, это отмечается в PROVENANCE.json и видно в интерфейсе.
set -euo pipefail

MB_SRC="${MB_SRC:-../MagicBrain}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/build/magicbrain-src"

if [ ! -d "$MB_SRC/.git" ]; then
  echo "нет git-репозитория MagicBrain по пути $MB_SRC" >&2
  exit 1
fi

SHA="$(git -C "$MB_SRC" rev-parse --short=12 HEAD)"
BRANCH="$(git -C "$MB_SRC" rev-parse --abbrev-ref HEAD)"
DESCRIBE="$(git -C "$MB_SRC" describe --tags --always --dirty 2>/dev/null || echo "$SHA")"
if [ -n "$(git -C "$MB_SRC" status --porcelain --untracked-files=no)" ]; then
  DIRTY="dirty"
else
  DIRTY="clean"
fi

rm -rf "$OUT"
mkdir -p "$OUT"
git -C "$MB_SRC" archive HEAD | tar -x -C "$OUT"

PLAY_SHA="$(git -C "$ROOT" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no 2>/dev/null)" ]; then
  PLAY_DIRTY="dirty"
else
  PLAY_DIRTY="clean"
fi
# Тег образа фиксируется один раз на сборку и переиспользуется, пока исходники
# не изменились. Раньше каждый вызов (в том числе неявный из `make up`) выдавал
# новый BUILD_ID: развёртывалось не то, что проверяли, а старые теги копились
# на общем разделе. Новый идентификатор появляется, только если сменился SHA
# MagicBrain или плейграунда, если любое из деревьев грязное (тогда содержимое
# может отличаться при том же SHA) или если его потребовали явно.
PREV_ID=""
PREV_SHA=""
PREV_PLAY=""
if [ -f "$ROOT/.env.build" ]; then
  PREV_ID="$(sed -n 's/^MBP_BUILD_ID=//p' "$ROOT/.env.build" | tail -1)"
  PREV_SHA="$(sed -n 's/^MBP_MB_CODE_SHA=//p' "$ROOT/.env.build" | tail -1)"
  PREV_PLAY="$(sed -n 's/^MBP_PLAYGROUND_SHA=//p' "$ROOT/.env.build" | tail -1)"
fi

if [ -n "${BUILD_ID:-}" ]; then
  REASON="задан снаружи"
elif [ "${MBP_FORCE_BUILD_ID:-0}" = "1" ]; then
  BUILD_ID="$(date -u +%Y%m%d-%H%M%S)-${SHA}"
  REASON="затребован новый (MBP_FORCE_BUILD_ID=1)"
elif [ "$DIRTY" = "dirty" ] || [ "$PLAY_DIRTY" = "dirty" ]; then
  BUILD_ID="$(date -u +%Y%m%d-%H%M%S)-${SHA}"
  REASON="рабочее дерево грязное, содержимое образа не определяется одним SHA"
elif [ -n "$PREV_ID" ] && [ "$PREV_SHA" = "$SHA" ] && [ "$PREV_PLAY" = "$PLAY_SHA" ]; then
  BUILD_ID="$PREV_ID"
  REASON="переиспользован: исходники не изменились"
else
  BUILD_ID="$(date -u +%Y%m%d-%H%M%S)-${SHA}"
  REASON="исходники изменились"
fi
cat > "$ROOT/build/PROVENANCE.json" <<JSON
{
  "magicbrain_source": "$MB_SRC",
  "magicbrain_sha": "$SHA",
  "magicbrain_branch": "$BRANCH",
  "magicbrain_describe": "$DESCRIBE",
  "magicbrain_worktree": "$DIRTY",
  "playground_sha": "$PLAY_SHA",
  "playground_worktree": "$PLAY_DIRTY",
  "build_id": "$BUILD_ID",
  "prepared_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON

cat > "$ROOT/.env.build" <<ENV
MBP_MB_CODE_SHA=$SHA
MBP_MB_CODE_DIRTY=$DIRTY
MBP_BUILD_ID=$BUILD_ID
MBP_PLAYGROUND_SHA=$PLAY_SHA
MBP_PLAYGROUND_DIRTY=$PLAY_DIRTY
ENV

echo "MagicBrain $SHA ($BRANCH, дерево $DIRTY) распакован в build/magicbrain-src"
echo "плейграунд $PLAY_SHA (дерево $PLAY_DIRTY)"
echo "build_id=$BUILD_ID ($REASON)"
