#!/bin/bash
set -e
cd /workspace/kalos
LOG=out/catalog-build.log
echo "start $(date -u)" > "$LOG"

# hero (dejong) ya existe: out/kalos-dejong-aurora.mp4
python3 render.py attractor-clifford 12 30 out/kalos-clifford-neon.mp4 >> "$LOG" 2>&1
python3 render.py harmonograph      12 30 out/kalos-harmonograph-ice.mp4 >> "$LOG" 2>&1

DEST=/workspace/media-kit/instagram/_galeria-multiproyecto/12-kalos-matematicas
mkdir -p "$DEST"
cp out/kalos-dejong-aurora.mp4 out/kalos-clifford-neon.mp4 out/kalos-harmonograph-ice.mp4 "$DEST"/

# poster por reel para el INDEX/mostrador
for f in "$DEST"/*.mp4; do
  ffmpeg -v error -y -i "$f" -vf "select=eq(n\,180)" -vframes 1 "${f%.mp4}-poster.jpg" >> "$LOG" 2>&1
done

cat > "$DEST/pies-instagram.md" <<'EOF'
# kalos — matemática bella en movimiento (pies IG)

Reels 1080x1920 nativos, generados por el repo `kalos` (github.com/stevenvo780/kalos).

## kalos-dejong-aurora.mp4 — Atractor de De Jong
> Dos ecuaciones. Infinitos caminos. Ninguno se repite, y aun así todos vuelven.
> Un atractor extraño: orden que emerge del caos determinista.
#matematicas #generativeart #strangeattractor #arte #ciencia

## kalos-clifford-neon.mp4 — Atractor de Clifford
> Millones de puntos siguiendo una regla mínima. Nadie los coordina;
> la forma aparece sola. Eso es emergencia, dibujada.
#matematicas #creativecoding #emergencia #arte #chaos

## kalos-harmonograph-ice.mp4 — Harmonógrafo
> Dos péndulos acoplados que se amortiguan despacio. La física dibujando
> sin mano. Una figura de Lissajous que respira y se apaga.
#matematicas #harmonograph #lissajous #arte #fisica
EOF

echo "done $(date -u)" >> "$LOG"
ls -la "$DEST" >> "$LOG"
