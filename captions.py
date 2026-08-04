#!/usr/bin/env python3
"""Genera el índice de captions/copys por reel de kalos.
Escribe out/captions.csv, out/captions.md y out/captions.json.
Cada entrada: scene, title, family, mp4, poster, caption(ES), hashtags."""
import csv, json, os

OUT = "out"
BASE_TAGS = "#matematicas #mathart #arteGenerativo #creativecoding #kalos #python #ciencia #reels"

# scene -> (title, family, caption_es, extra_tags)
C = {
"harmonograph": ("Harmonógrafo", "Curvas / osciladores",
 "Dos péndulos acoplados que se amortiguan lentamente tejen figuras de Lissajous. Cada vuelta pierde un poco de energía y la curva se cierra sobre sí misma como un bordado de luz.",
 "#harmonograph #lissajous #oscilador"),
"attractor-dejong": ("Atractor de De Jong", "Atractores extraños",
 "Cuatro senos iterados millones de veces. Cada punto cae donde el sistema lo empuja y, sumados, dibujan un velo imposible que nunca se repite.",
 "#strangeattractor #dejong #caos"),
"attractor-clifford": ("Atractor de Clifford", "Atractores extraños",
 "El mapa de Clifford: dos ecuaciones, infinitas órbitas. El caos determinista pintado como filamentos de neón sobre el vacío.",
 "#strangeattractor #clifford #caos"),
"attractor-lorenz": ("Atractor de Lorenz", "Atractores extraños",
 "La mariposa que le puso nombre al efecto mariposa. El modelo de convección de Lorenz: orden dentro del caos, nunca dos vueltas iguales.",
 "#lorenz #butterflyeffect #caos"),
"attractor-aizawa": ("Atractor de Aizawa", "Atractores extraños",
 "Una órbita que se enrosca sobre un toro y escapa por los polos. Aizawa convierte tres ecuaciones en una escultura que respira.",
 "#aizawa #strangeattractor #caos"),
"attractor-rossler": ("Atractor de Rössler", "Atractores extraños",
 "El atractor más simple que aún es caótico: una espiral que se estira y se pliega una y otra vez. Rössler lo diseñó casi como un juguete.",
 "#rossler #caos #dynamics"),
"attractor-thomas": ("Atractor de Thomas", "Atractores extraños",
 "Simetría cíclica perfecta: las mismas ecuaciones en x, y, z. El resultado es un laberinto ciclado que parece respirar en las tres direcciones a la vez.",
 "#thomasattractor #caos #simetria"),
"attractor-halvorsen": ("Atractor de Halvorsen", "Atractores extraños",
 "Tres ejes, una regla cíclica y realimentación cuadrática. Halvorsen enrosca el espacio en una concha de caracol de luz.",
 "#halvorsen #strangeattractor #caos"),
"attractor-dadras": ("Atractor de Dadras", "Atractores extraños",
 "Un atractor que se abre en alas y lazos entrelazados. Pura dinámica no lineal graficada como filigrana.",
 "#dadras #strangeattractor #caos"),
"attractor-chen": ("Atractor de Chen", "Atractores extraños",
 "Primo del de Lorenz, pero más retorcido. El sistema de Chen dobla el flujo hasta formar un doble scroll incandescente.",
 "#chenattractor #caos #dynamics"),
"attractor-gumowski-mira": ("Atractor de Gumowski-Mira", "Atractores extraños",
 "Nacido en el CERN para estudiar haces de partículas. Con un pequeño ajuste explota en criaturas simétricas que parecen vida marina.",
 "#gumowskimira #cern #strangeattractor"),
"attractor-hopalong": ("Atractor Hopalong", "Atractores extraños",
 "El 'salta que salta' de Barry Martin: una raíz cuadrada con signo y millones de brincos que se acumulan en anillos de interferencia.",
 "#hopalong #strangeattractor #caos"),
"attractor-ikeda": ("Mapa de Ikeda", "Atractores extraños",
 "Modela la luz atrapada en una cavidad óptica no lineal. La rotación dependiente de la posición enrolla el plano en una espiral láser.",
 "#ikeda #optica #caos"),
"attractor-tinkerbell": ("Mapa de Tinkerbell", "Atractores extraños",
 "Un mapa cuadrático mínimo con nombre de hada. Traza un ala curva que se rellena de detalle fractal a medida que itera.",
 "#tinkerbell #strangeattractor #caos"),
"attractor-svensson": ("Atractor de Svensson", "Atractores extraños",
 "Variante de la familia Peter de Jong: senos y cosenos anidados que tejen una malla de encaje sobre el negro.",
 "#svensson #strangeattractor #encaje"),
"attractor-bedhead": ("Atractor Bedhead", "Atractores extraños",
 "Dos líneas de código, un caos peinado al despertar. El 'pelo de almohada' de la matemática recreativa.",
 "#bedhead #strangeattractor #caos"),
"attractor-duffing": ("Oscilador de Duffing", "Atractores extraños",
 "Un resorte no lineal forzado que nunca decide su ritmo. El atractor de Duffing pliega el espacio de fases en cintas caóticas.",
 "#duffing #oscilador #caos"),
"attractor-standard-map": ("Mapa estándar (Chirikov)", "Atractores extraños",
 "El puente entre orden y caos: islas de estabilidad flotando en un mar estocástico. Fundamental en física del caos hamiltoniano.",
 "#chirikov #standardmap #caoshamiltoniano"),
"fractal-mandelbrot": ("Conjunto de Mandelbrot", "Fractales",
 "z ↦ z² + c, iterado hasta el infinito. El objeto más famoso de las matemáticas: frontera infinita atrapada en un área finita. Aquí, un zoom sin fondo.",
 "#mandelbrot #fractal #infinito"),
"fractal-julia": ("Conjuntos de Julia", "Fractales",
 "La misma regla que Mandelbrot, pero fijando c y moviendo el mundo. Cada c es un Julia distinto; aquí morfean uno en otro sin costura.",
 "#julia #fractal #dynamics"),
"fractal-burning-ship": ("Fractal Burning Ship", "Fractales",
 "Toma valor absoluto antes de elevar al cuadrado y el Mandelbrot se incendia: un barco en llamas emergiendo de la niebla compleja.",
 "#burningship #fractal #complejo"),
"fractal-newton": ("Fractal de Newton", "Fractales",
 "El método de Newton para hallar raíces, coloreado por a cuál converge cada punto. Las fronteras entre cuencas son infinitamente entrelazadas.",
 "#newtonfractal #fractal #raices"),
"fractal-lyapunov": ("Fractal de Lyapunov (Markus)", "Fractales",
 "Mapa logístico alternando dos ritmos de reproducción. El exponente de Lyapunov pinta la estabilidad: bordes que parecen tejido biológico.",
 "#lyapunov #fractal #caos"),
"fractal-phoenix": ("Fractal Phoenix", "Fractales",
 "Un Julia con memoria: cada iteración recuerda la anterior. Nacen zarcillos que se enroscan como plumas de ave fénix.",
 "#phoenixfractal #fractal #complejo"),
"fractal-multibrot": ("Multibrot morfante", "Fractales",
 "z ↦ zⁿ + c con n variando en el tiempo. El Mandelbrot crece pétalos: de 2 lóbulos a flores de simetría cambiante.",
 "#multibrot #fractal #morphing"),
"life-conway": ("Juego de la Vida de Conway", "Vida / autómatas",
 "Cuatro reglas, un universo. Nace, sobrevive o muere según sus vecinos. Con estela decadente, las naves y osciladores dejan rastros de cometa.",
 "#gameoflife #conway #automata"),
"life-lenia": ("Lenia", "Vida / autómatas",
 "El Juego de la Vida hecho continuo y suave. Emergen 'criaturas' orgánicas que nadan, se dividen y giran: vida artificial de verdad.",
 "#lenia #artificiallife #automata"),
"life-smoothlife": ("SmoothLife", "Vida / autómatas",
 "La versión continua de Conway en espacio y tiempo. Los gliders se vuelven gotas fluidas que se persiguen por el plano.",
 "#smoothlife #artificiallife #automata"),
"ca-brians-brain": ("Brian's Brain", "Vida / autómatas",
 "Un autómata de tres estados: vivo, moribundo, muerto. El plano se llena de naves que explotan en frentes de onda perpetuos.",
 "#briansbrain #automata #cellular"),
"ca-cyclic": ("Autómata cíclico", "Vida / autómatas",
 "Cada color solo puede ceder al siguiente en el ciclo. Del ruido nacen espirales que se organizan solas: orden desde el caos puro.",
 "#cyclicca #automata #espirales"),
"ca-rule110": ("Regla 110", "Vida / autómatas",
 "Un autómata 1D de una sola línea, capaz de computación universal. Su scroll teje triángulos anidados: Turing-completo en blanco y negro.",
 "#rule110 #turingcomplete #automata"),
"ca-rule30": ("Regla 30", "Vida / autómatas",
 "Reglas triviales, salida caótica. Wolfram la usó como generador de números aleatorios: el desorden brotando del determinismo.",
 "#rule30 #wolfram #automata"),
"ca-langtons-ant": ("Hormiga de Langton", "Vida / autómatas",
 "Una hormiga, dos reglas. Diez mil pasos de caos aparente y de pronto —sin avisar— construye una 'autopista' periódica para siempre.",
 "#langtonsant #automata #emergencia"),
"reaction-gray-scott": ("Reacción-difusión Gray-Scott", "Patrones / PDE",
 "Dos químicos que se alimentan y se matan. La misma ecuación que pinta manchas de leopardo, laberintos y mitosis según dos números.",
 "#grayscott #reactiondiffusion #turing"),
"reaction-fitzhugh-nagumo": ("FitzHugh-Nagumo", "Patrones / PDE",
 "El modelo simplificado del impulso nervioso. Ondas de excitación recorren el tejido como late un corazón: frentes que se persiguen.",
 "#fitzhughnagumo #excitablemedia #pde"),
"reaction-turing": ("Patrones de Turing", "Patrones / PDE",
 "La idea de Alan Turing sobre cómo un embrión decide sus manchas y rayas. Química pura organizándose en piel animal.",
 "#turingpatterns #morfogenesis #pde"),
"reaction-belousov-zhabotinsky": ("Reacción Belousov-Zhabotinsky", "Patrones / PDE",
 "La reacción química que oscila y no se detiene. Espirales que giran solas en un medio excitable: relojes químicos de verdad.",
 "#belousovzhabotinsky #espirales #quimica"),
"reaction-swift-hohenberg": ("Swift-Hohenberg", "Patrones / PDE",
 "La ecuación de la formación de patrones por excelencia. Selecciona una longitud de onda favorita y llena el plano de rayas y hexágonos.",
 "#swifthohenberg #patterns #pde"),
"growth-physarum": ("Physarum (moho mucilaginoso)", "Crecimiento / bio",
 "El moho sin cerebro que resuelve laberintos. Miles de agentes depositan y siguen rastros hasta tejer una red de micelio viviente.",
 "#physarum #slimemold #micelio"),
"growth-dla": ("Agregación limitada por difusión (DLA)", "Crecimiento / bio",
 "Partículas que vagan al azar y se pegan al tocar. Así crece el coral, la escarcha y el rayo: dendritas fractales desde el borde.",
 "#dla #dendritas #fractal"),
"growth-eden": ("Crecimiento de Eden", "Crecimiento / bio",
 "El modelo más simple de una colonia que crece: cada celda del borde puede reproducirse. Una mancha que avanza con frontera rugosa universal.",
 "#edengrowth #colonia #crecimiento"),
"growth-barnsley-fern": ("Helecho de Barnsley", "Crecimiento / bio",
 "Cuatro transformaciones afines aplicadas al azar dibujan un helecho perfecto. La prueba de que el azar más simple guarda una planta entera.",
 "#barnsleyfern #ifs #fractal"),
"growth-lsystem-plant": ("Planta L-system", "Crecimiento / bio",
 "Una gramática que se reescribe a sí misma, letra a letra, hasta ramificar en un árbol. El lenguaje formal como semilla botánica.",
 "#lsystem #botanica #fractal"),
"spread-sir": ("Epidemia SIR en malla", "Propagación / física",
 "Susceptible, infectado, recuperado. La misma matemática de una epidemia real, corriendo como ondas de contagio sobre una red.",
 "#sirmodel #epidemia #contagio"),
"spread-forest-fire": ("Incendio forestal", "Propagación / física",
 "Árboles que crecen, rayos que caen, fuego que salta al vecino. Un modelo clásico de criticalidad auto-organizada, en llamas.",
 "#forestfire #criticalidad #modelo"),
"spread-percolation": ("Percolación", "Propagación / física",
 "Abre poros al azar hasta que —de golpe— un camino cruza todo. La transición de fase que explica desde el café hasta las epidemias.",
 "#percolation #transicionfase #cluster"),
"physics-wave-interference": ("Interferencia de ondas", "Propagación / física",
 "Varias fuentes emitiendo a la vez. Donde las crestas se suman brilla; donde se cancelan, oscuridad. El principio de superposición hecho luz.",
 "#interferencia #ondas #fisica"),
"physics-chladni": ("Figuras de Chladni (cimática)", "Propagación / física",
 "Una placa que vibra y la arena huye de los nodos. Los modos propios de una membrana dibujados como mandalas sonoros.",
 "#chladni #cimatica #resonancia"),
"physics-double-pendulum": ("Péndulo doble (abanico de caos)", "Propagación / física",
 "Un péndulo colgado de otro: el ejemplo de libro del caos. Un abanico de ellos, casi idénticos, que divergen y se separan sin remedio.",
 "#doublependulum #caos #fisica"),
"physics-nbody": ("Gravedad de N cuerpos", "Propagación / física",
 "Masas que se atraen todas con todas. La danza gravitatoria que forma galaxias, simulada como estelas de estrellas.",
 "#nbody #gravedad #astrofisica"),
"physics-curl-flow": ("Campo de flujo (curl / Perlin)", "Propagación / física",
 "Miles de partículas arrastradas por un campo vectorial suave. Así se ve el viento, el humo y el agua: líneas de corriente vivas.",
 "#flowfield #curlnoise #generativo"),
"physics-ising": ("Modelo de Ising", "Propagación / física",
 "Espines magnéticos que quieren imitar a sus vecinos contra el ruido térmico. En el punto crítico, dominios de todos los tamaños a la vez.",
 "#isingmodel #magnetismo #transicionfase"),
"pattern-phyllotaxis": ("Filotaxis (girasol)", "Curvas / patrones",
 "El ángulo áureo, 137,5°, colocando semilla tras semilla. Así empaqueta un girasol sus pipas: espirales de Fibonacci en acción.",
 "#filotaxis #anguloaureo #fibonacci"),
"curve-maurer-rose": ("Rosas de Maurer", "Curvas / patrones",
 "Une puntos de una rosa polar con líneas rectas y aparecen redes de estrellas que la curva suave escondía. Geometría oculta a plena vista.",
 "#maurerrose #curvapolar #geometria"),
"curve-fourier-epicycles": ("Epiciclos de Fourier", "Curvas / patrones",
 "Círculos girando sobre círculos girando sobre círculos. La serie de Fourier: cualquier trazo del mundo reconstruido con puro giro.",
 "#fourier #epiciclos #series"),
}

# orden canónico del catálogo (según registry)
ORDER = [
 "attractor-dejong","attractor-clifford","harmonograph","attractor-lorenz","attractor-aizawa",
 "attractor-rossler","attractor-thomas","attractor-halvorsen","attractor-dadras","attractor-chen",
 "attractor-gumowski-mira","attractor-hopalong","attractor-ikeda","attractor-tinkerbell","attractor-svensson",
 "attractor-bedhead","attractor-duffing","attractor-standard-map","fractal-mandelbrot","fractal-julia",
 "fractal-burning-ship","fractal-newton","fractal-lyapunov","fractal-phoenix","fractal-multibrot",
 "life-conway","life-lenia","life-smoothlife","ca-brians-brain","ca-cyclic","ca-rule110","ca-rule30",
 "ca-langtons-ant","reaction-gray-scott","reaction-fitzhugh-nagumo","reaction-turing",
 "reaction-belousov-zhabotinsky","reaction-swift-hohenberg","growth-physarum","growth-dla","growth-eden",
 "growth-barnsley-fern","growth-lsystem-plant","spread-sir","spread-forest-fire","spread-percolation",
 "physics-wave-interference","physics-chladni","physics-double-pendulum","physics-nbody","physics-curl-flow",
 "pattern-phyllotaxis","curve-maurer-rose","curve-fourier-epicycles","physics-ising",
]

assert set(ORDER) == set(C), f"desajuste: {set(ORDER)^set(C)}"

rows = []
for scene in ORDER:
    title, family, cap, extra = C[scene]
    mp4 = f"{scene}.mp4"
    poster = f"posters/{scene}.png"
    have_mp4 = os.path.exists(os.path.join(OUT, mp4))
    have_poster = os.path.exists(os.path.join(OUT, poster))
    tags = f"{BASE_TAGS} {extra}".strip()
    rows.append({
        "scene": scene, "title": title, "family": family,
        "mp4": mp4, "poster": poster, "caption": cap, "hashtags": tags,
        "mp4_ok": have_mp4, "poster_ok": have_poster,
    })

# CSV
with open(os.path.join(OUT, "captions.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["scene","title","family","mp4","poster","caption","hashtags"])
    w.writeheader()
    for r in rows:
        w.writerow({k: r[k] for k in w.fieldnames})

# JSON
with open(os.path.join(OUT, "captions.json"), "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

# MD
with open(os.path.join(OUT, "captions.md"), "w", encoding="utf-8") as f:
    f.write("# kalos — índice de captions por reel\n\n")
    f.write(f"{len(rows)} reels listos para publicar. Cada bloque: título, archivos (mp4 + póster), copy en español y hashtags.\n\n")
    f.write("Uso: en Instagram, subir el `.mp4` como reel; el `.png` de posters sirve de portada/miniatura.\n\n")
    fam = None
    for i, r in enumerate(rows, 1):
        if r["family"] != fam:
            fam = r["family"]
            f.write(f"\n## {fam}\n\n")
        f.write(f"### {i}. {r['title']}  \n")
        f.write(f"`{r['mp4']}` · portada `{r['poster']}`  \n\n")
        f.write(f"{r['caption']}\n\n")
        f.write(f"{r['hashtags']}\n\n")
        f.write("---\n\n")

missing = [r["scene"] for r in rows if not (r["mp4_ok"] and r["poster_ok"])]
print(f"reels={len(rows)} csv+md+json escritos en out/")
print(f"faltantes(mp4/poster)={len(missing)}: {missing if missing else 'ninguno'}")
