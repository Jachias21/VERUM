# VERUM Landing Page — Instrucciones de construcción

## Contexto
Landing page promocional del proyecto VERUM, un bot de Telegram para verificación forense de imágenes y textos virales. La web presenta el producto al usuario final y al tribunal del TFM. El diseño sigue la filosofía de Duolingo: la mascota es el hilo conductor visual que cambia de pose según la sección mientras el usuario hace scroll.

**Stack:** Next.js (App Router) + Tailwind CSS + Framer Motion  
**Deploy:** Vercel  
**Bot CTA:** `https://t.me/Verum_tfm_bot`  
**Idiomas:** Bilingüe ES / EN con selector en la navbar

---

## Paso 0 — Assets

Copiar los siguientes PNGs a `public/mascot/` con estos nombres exactos:

| Nombre destino | Archivo fuente |
|---|---|
| `verum-saludando.png` | VERUM_saludando.png |
| `verum-lupa.png` | VERUM_lupa.png |
| `verum-focus.png` | VERUM_focus.png |
| `verum-pensando.png` | Gemini_Generated_Image_bu7pbb...png |
| `verum-escudo.png` | Gemini_Generated_Image_c10pj6...png |

Copiar también `Logo_2.png` a `public/logo.png`.

---

## Estructura de archivos

```
landing/
├── public/
│   ├── logo.png
│   └── mascot/
│       ├── verum-saludando.png
│       ├── verum-lupa.png
│       ├── verum-focus.png
│       ├── verum-pensando.png
│       └── verum-escudo.png
├── src/
│   └── app/
│       ├── layout.tsx
│       ├── page.tsx
│       ├── globals.css
│       └── components/
│           ├── Navbar.tsx
│           ├── Hero.tsx
│           ├── WhatIsVerum.tsx
│           ├── HowItWorks.tsx
│           ├── WhyVerum.tsx
│           └── CTAFinal.tsx
```

---

## globals.css

Importar desde Google Fonts:
- **Poppins** pesos 600 y 700
- **Nunito** pesos 400 y 500

Definir las siguientes variables CSS en `:root`:

```
--color-navy: #0D1B2A
--color-teal: #4CCAD1
--color-teal-light: #BFE8EE
--color-gray-light: #E6E8ED
--color-white: #FFFFFF
--color-text-primary: #0D1B2A
--color-text-secondary: #4A5568
```

Aplicar `Nunito` como `font-family` base en `body`. Aplicar `Poppins` en todos los `h1, h2, h3`.

Definir el keyframe de animación flotante del personaje:

```
@keyframes float {
  0%   { transform: translateY(0px); }
  50%  { transform: translateY(-8px); }
  100% { transform: translateY(0px); }
}
```

Con duración de 3s, easing `ease-in-out`, repetición `infinite`. Crear una clase utilitaria `.animate-float` que aplique esta animación.

---

## layout.tsx

- Metadata: título `VERUM — Asistente de Verdad`, descripción `Tu perito forense de bolsillo contra las Fake News`
- Importar `globals.css`
- Renderizar `{children}` envuelto en `<main>`

---

## Gestión del idioma

Crear un contexto React `LanguageContext` en `src/app/context/LanguageContext.tsx` que exponga:
- `lang`: valor `"es"` o `"en"`, inicializado a `"es"`
- `setLang`: función para alternar entre los dos

Cada componente importa el contexto y usa un objeto local `t` con las traducciones:

```ts
const translations = {
  es: { title: "...", subtitle: "..." },
  en: { title: "...", subtitle: "..." },
}
const { lang } = useLanguage()
const t = translations[lang]
```

---

## Navbar.tsx

**Comportamiento:**
- Posición fija arriba (`fixed top-0`)
- Fondo blanco con sombra suave que aparece únicamente cuando `window.scrollY > 10` — implementar con `useState` + `useEffect`
- Z-index alto para estar siempre encima del contenido

**Contenido izquierda:**
- `<Image>` de `public/logo.png` con altura 36px
- Texto "VERUM" en Poppins 700, color navy, tamaño `text-xl`

**Contenido derecha:**
- Link ancla `"Cómo funciona"` / `"How it works"` → `#how`
- Link ancla `"Por qué VERUM"` / `"Why VERUM"` → `#why`
- Selector de idioma: dos botones `ES` y `EN` en Poppins 600, tamaño pequeño. El activo en teal con fondo teal suave, el inactivo en gris. Al hacer clic llaman a `setLang`
- Botón CTA: `"Abrir en Telegram"` / `"Open in Telegram"` — pill redondeado (`rounded-full`), fondo teal `#4CCAD1`, texto navy, `font-weight 600`. Hover: `#3BB5BC`. Link externo a `https://t.me/Verum_tfm_bot`

**Mobile:** ocultar los links de navegación y el selector de idioma en mobile. Mantener visible solo el logo y el botón CTA.

---

## Hero.tsx

**Fondo:** blanco puro.

Añadir un blob decorativo detrás del personaje: `div` con posición absoluta en esquina superior derecha, `border-radius: 50%`, dimensiones ~500x500px, color `#BFE8EE`, opacidad 30%, `filter: blur(80px)`. `pointer-events: none`.

**Layout desktop:** dos columnas (`grid grid-cols-2`), gap generoso. En mobile: una columna, personaje debajo del texto.

**Columna izquierda — de arriba a abajo:**

1. Eyebrow badge: texto `"ASISTENTE DE VERDAD"` / `"TRUTH ASSISTANT"` — pill con `border` de 1px en teal, texto teal, fondo `#BFE8EE` con opacidad 40%, texto `text-xs`, `tracking-widest`, `uppercase`, `font-semibold`

2. H1 en Poppins 700, tamaño grande (`text-5xl md:text-6xl`), color navy, texto: `"Analiza. Verifica. Explica."`

3. Párrafo Nunito 400, color text-secondary, `text-lg`: `"Tu aliado contra la desinformación en mensajería privada. Antes de compartir, consulta."` / `"Your ally against disinformation in private messaging. Before sharing, verify."`

4. Fila de dos botones con gap:
   - **Botón primario:** `"Hablar con VERUM"` / `"Talk to VERUM"` — pill, fondo teal, texto navy, Poppins 600. Dentro a la derecha: pequeño círculo navy con `→` blanco dentro. Link a `https://t.me/Verum_tfm_bot`. Hover: escala leve `scale-[1.02]`, transición `cubic-bezier(0.32, 0.72, 0, 1)`
   - **Botón secundario:** `"Cómo funciona"` / `"How it works"` — pill, fondo transparente, borde navy, texto navy. Ancla a `#how`

**Columna derecha:**
- `<Image>` de `public/mascot/verum-saludando.png` con `width: 420px`, `height: 420px`, `object-fit: contain`
- Aplicar clase `.animate-float`
- Envolver en Framer Motion `motion.div` con animación de entrada: `initial={{ opacity: 0, y: 40 }}` → `animate={{ opacity: 1, y: 0 }}` con `transition={{ type: "spring", duration: 0.8, delay: 0.3 }}`

**Animaciones de entrada del texto:** envolver cada elemento del lado izquierdo en `motion.div` con `initial={{ opacity: 0, y: 20 }}` → `animate={{ opacity: 1, y: 0 }}`. Stagger manual: eyebrow delay 0s, h1 delay 0.1s, párrafo delay 0.2s, botones delay 0.3s.

---

## WhatIsVerum.tsx — `id="what"`

**Fondo:** `#F7F8FA` (blanco roto muy suave)

**Layout:** centrado con `max-width` contenido, padding vertical generoso (`py-24`)

**Contenido superior centrado:**
- Eyebrow: `"¿QUIÉN ES VERUM?"` / `"WHO IS VERUM?"`
- H2 Poppins 700 `text-4xl` centrado: `"En un mundo donde la información se manipula, VERUM transforma datos en evidencia para revelar la verdad"` — la palabra `VERUM` en color teal
- Párrafo Nunito centrado, max-width ~65ch: `"VERUM es un asistente digital inteligente especializado en verificar la autenticidad de imágenes y textos que circulan en mensajería privada."` / `"VERUM is an intelligent digital assistant specialized in verifying the authenticity of images and texts circulating in private messaging."`

**Grid de 4 cards** debajo (`grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4`), gap generoso:

Cada card:
- Fondo blanco
- Border-radius generoso (`rounded-2xl`)
- Sombra suave (`shadow-sm`)
- Borde superior de 3px color teal
- Padding interno `p-6`
- Icono SVG inline (24x24px, color teal) — sin librerías externas, SVG paths simples
- Título Poppins 600 navy
- Descripción Nunito 400 text-secondary

| Icono | Título ES | Título EN | Descripción ES | Descripción EN |
|---|---|---|---|---|
| Lupa | Analiza | Analyze | Examina el contenido en profundidad | Examines content in depth |
| Escudo check | Verifica | Verify | Contrasta con múltiples fuentes y evidencias | Cross-checks with multiple sources |
| Burbuja texto | Explica | Explain | Te muestra el resultado y el porqué | Shows you the result and the why |
| Corazón/escudo | Protege | Protect | Te ayuda a tomar decisiones informadas | Helps you make informed decisions |

**Animaciones:** cada card con Framer Motion `whileInView`, `viewport={{ once: true }}`, `initial={{ opacity: 0, y: 30 }}` → `whileInView={{ opacity: 1, y: 0 }}`. Stagger de 0.1s entre cards (`delay: index * 0.1`).

---

## HowItWorks.tsx — `id="how"`

**Fondo:** blanco

**Padding vertical:** muy generoso (`py-32`)

**Header centrado:**
- Eyebrow: `"CÓMO FUNCIONA"` / `"HOW IT WORKS"`
- H2: `"Cuatro pasos, un veredicto"` / `"Four steps, one verdict"`

**Layout desktop:** dos columnas (`grid grid-cols-2`). Columna izquierda: los 4 steps en scroll. Columna derecha: personaje sticky.

Columna derecha: `position: sticky`, `top: 120px`, centrada verticalmente. Contiene un `<Image>` cuyo `src` cambia dinámicamente según el step activo. Envolver el `<Image>` en `AnimatePresence` de Framer Motion para que al cambiar el `src` haga fade out del anterior y fade in del nuevo (`initial={{ opacity: 0, scale: 0.95 }}` → `animate={{ opacity: 1, scale: 1 }}` → `exit={{ opacity: 0, scale: 0.95 }}`). Transición 0.3s.

**Detección del step activo:** usar `useRef` en cada step + `IntersectionObserver` con `threshold: 0.6`. Cuando un step entra al viewport, actualizar el `useState` del personaje activo.

**Cada step en la columna izquierda:**
- Padding vertical `py-20` para dar espacio al scroll
- Número del step (`01`, `02`, `03`, `04`) en Poppins 700, tamaño `text-8xl`, color `#BFE8EE` (teal muy claro), posición relativa como decoración de fondo detrás del título
- Título Poppins 700 `text-3xl` navy encima del número decorativo
- Descripción Nunito `text-lg` text-secondary

| Step | Personaje | Título ES | Título EN | Descripción ES | Descripción EN |
|---|---|---|---|---|---|
| 01 | verum-saludando.png | Envías el contenido | You send the content | Manda la imagen o texto sospechoso al bot de Telegram | Send the suspicious image or text to the Telegram bot |
| 02 | verum-lupa.png | VERUM analiza | VERUM analyzes | Aplica visión forense y análisis de frecuencias para detectar manipulación | Applies forensic vision and frequency analysis to detect manipulation |
| 03 | verum-focus.png | Verifica las fuentes | Verifies sources | Cruza los datos con bases de verificación oficiales y artículos de fact-checking | Cross-references data with official verification databases |
| 04 | verum-escudo.png | Recibes el veredicto | You get the verdict | Respuesta clara, explicada y con fuentes en menos de 15 segundos | Clear, explained response with sources in under 15 seconds |

**Mobile:** ocultar el layout sticky. Mostrar cada step en columna única con el personaje correspondiente encima del texto de cada step, tamaño reducido (~200px).

---

## WhyVerum.tsx — `id="why"`

**Fondo:** navy `#0D1B2A` — única sección oscura de la página

**Padding vertical:** `py-24`

**Layout desktop:** dos columnas. Izquierda texto, derecha personaje.

**Columna izquierda — textos:**
- Eyebrow en teal: `"EL PROBLEMA"` / `"THE PROBLEM"`
- H2 en blanco Poppins 700 `text-4xl`: `"La desinformación más peligrosa viaja en privado"` / `"The most dangerous disinformation travels privately"`
- Párrafo en gris claro Nunito: explicar el concepto de Dark Social — WhatsApp, Telegram, cifrado extremo a extremo, sin moderación algorítmica. 2-3 líneas.

3 bullet points problema con icono `✕` en `#FF6B6B` (rojo suave):
- ES: `Sin herramientas de verificación nativas` / EN: `No native verification tools`
- ES: `Contenido sintético indetectable al ojo humano` / EN: `Synthetic content undetectable to the human eye`
- ES: `Propagación exponencial antes de ser detectada` / EN: `Exponential spread before detection`

3 bullet points solución con icono `✓` en teal:
- ES: `Análisis forense de imágenes con IA explicable` / EN: `Forensic image analysis with explainable AI`
- ES: `Verificación contra bases de fact-checking oficiales` / EN: `Verification against official fact-checking databases`
- ES: `Privacidad total — tus datos nunca se almacenan` / EN: `Total privacy — your data is never stored`

**Columna derecha:**
- `<Image>` de `public/mascot/verum-pensando.png`, tamaño ~380px
- Framer Motion `whileInView`: `initial={{ opacity: 0, x: 40 }}` → `whileInView={{ opacity: 1, x: 0 }}`, `viewport={{ once: true }}`, `transition={{ duration: 0.7 }}`

---

## CTAFinal.tsx

**Fondo:** gradiente vertical de blanco `#FFFFFF` a teal claro `#BFE8EE`

**Layout:** columna única centrada, `py-32`

**Contenido de arriba a abajo:**
1. `<Image>` de `public/mascot/verum-escudo.png`, tamaño ~280px, centrada, con clase `.animate-float`
2. H2 Poppins 700 `text-4xl` navy centrado: `"¿Listo para verificar la verdad?"` / `"Ready to verify the truth?"`
3. Párrafo Nunito `text-lg` text-secondary centrado: `"Únete a VERUM en Telegram. Gratis, privado y disponible 24/7"` / `"Join VERUM on Telegram. Free, private and available 24/7"`
4. Botón primario idéntico al del Hero: `"Abrir VERUM en Telegram"` / `"Open VERUM on Telegram"` → `https://t.me/Verum_tfm_bot`
5. Texto pequeño Nunito `text-sm` en gris: `"Sin registro. Sin datos personales. RGPD compliant."` / `"No sign-up. No personal data. GDPR compliant."`

**Footer** dentro de este componente, separado por un `border-top` gris claro, `mt-16 pt-8`:
- Centrado
- Texto `© 2026 VERUM · TFM Máster IA & Big Data`
- Links separados por `·`: `Privacidad` / `Privacy` y `GitHub`
- Todo en Nunito `text-sm` text-secondary

---

## page.tsx

Importar y renderizar los componentes en este orden exacto:

```
<Navbar />
<Hero />
<WhatIsVerum />
<HowItWorks />
<WhyVerum />
<CTAFinal />
```

Añadir `scroll-behavior: smooth` en el elemento `<html>` via `globals.css` para que las anclas de navegación funcionen suavemente.

---

## Notas finales para el agente

1. **Framer Motion:** usar siempre `whileInView` con `viewport={{ once: true }}` para que las animaciones de entrada no se repitan al hacer scroll hacia arriba

2. **Imágenes:** usar siempre el componente `<Image>` de Next.js con `alt` descriptivo. Para el mascot, usar `priority` en el Hero ya que es above the fold

3. **Links externos:** todos los links a Telegram deben tener `target="_blank"` y `rel="noopener noreferrer"`

4. **AnimatePresence:** importar de `framer-motion`. El cambio de personaje en HowItWorks necesita una `key` única en el `motion.div` interior para que Framer Motion detecte el cambio y ejecute la animación de salida/entrada correctamente. Usar el nombre del archivo del personaje como `key`

5. **IntersectionObserver en HowItWorks:** crear un array de `useRef` para los 4 steps. En un `useEffect`, registrar un observer por cada ref. Al desmontar el componente, llamar a `observer.disconnect()` para evitar memory leaks

6. **Mobile first:** usar breakpoints Tailwind `md:` y `lg:` para activar layouts de columnas. Por debajo de `md` todo es columna única

7. **No usar `h-screen`:** usar siempre `min-h-[100dvh]` si se necesita altura de viewport completa, para evitar el jumping del viewport en Safari iOS

8. **Rendimiento:** el blob decorativo del Hero debe tener `pointer-events-none` y `aria-hidden="true"` para no interferir con accesibilidad ni eventos de click
