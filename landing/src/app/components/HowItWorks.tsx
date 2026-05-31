"use client";

import { useState, useEffect, useRef } from "react";
import Image from "next/image";
import { motion, AnimatePresence, useScroll } from "framer-motion";
import { useLanguage } from "../context/LanguageContext";

/* ─── Translations ───────────────────────────────────────────────── */
const translations = {
  es: {
    eyebrow: "CÓMO FUNCIONA",
    heading: "Cuatro pasos, un veredicto",
  },
  en: {
    eyebrow: "HOW IT WORKS",
    heading: "Four steps, one verdict",
  },
};

/* ─── Steps data ─────────────────────────────────────────────────── */
const stepsData = [
  {
    number: "01",
    mascot: "/mascot/verum-saludando.png",
    titleEs: "Envías el contenido",
    titleEn: "You send the content",
    descEs:
      "Manda la imagen o texto sospechoso directamente al bot de Telegram. Sin apps adicionales, sin registro previo.",
    descEn:
      "Send the suspicious image or text directly to the Telegram bot. No extra apps, no prior registration.",
    contextEs: "Formatos soportados",
    contextEn: "Supported formats",
    tags: ["JPG / PNG", "Texto", "Cadenas virales"],
    tagsEn: ["JPG / PNG", "Text", "Viral chains"],
  },
  {
    number: "02",
    mascot: "/mascot/verum-lupa.png",
    titleEs: "VERUM analiza",
    titleEn: "VERUM analyzes",
    descEs:
      "Aplica visión forense con Transformada de Fourier y análisis de ruido de sensor para detectar artefactos sintéticos invisibles al ojo humano.",
    descEn:
      "Applies forensic vision with Fourier Transform and sensor noise analysis to detect synthetic artifacts invisible to the human eye.",
    contextEs: "Motores activos",
    contextEn: "Active engines",
    tags: ["Fourier DFT", "PRNU", "CNN Two-Stream"],
    tagsEn: ["Fourier DFT", "PRNU", "CNN Two-Stream"],
  },
  {
    number: "03",
    mascot: "/mascot/verum-focus.png",
    titleEs: "Verifica las fuentes",
    titleEn: "Verifies sources",
    descEs:
      "Cruza las entidades extraídas con bases de fact-checking oficiales. Maldita.es, Newtral, Verificat y Google Fact Check en tiempo real.",
    descEn:
      "Cross-references extracted entities with official fact-checking databases. Maldita.es, Newtral, Verificat and Google Fact Check in real time.",
    contextEs: "Fuentes consultadas",
    contextEn: "Sources checked",
    tags: ["Maldita.es", "Newtral", "Google Fact Check"],
    tagsEn: ["Maldita.es", "Newtral", "Google Fact Check"],
  },
  {
    number: "04",
    mascot: "/mascot/verum-escudo.png",
    titleEs: "Recibes el veredicto",
    titleEn: "You get the verdict",
    descEs:
      "Respuesta clara, explicada y con fuentes en menos de 15 segundos. Con mapa de calor Grad-CAM si es una imagen manipulada.",
    descEn:
      "Clear, explained response with sources in under 15 seconds. With Grad-CAM heatmap if the image is manipulated.",
    contextEs: "Tiempo de respuesta",
    contextEn: "Response time",
    tags: ["< 15s imagen", "< 5s texto", "Grad-CAM"],
    tagsEn: ["< 15s image", "< 5s text", "Grad-CAM"],
  },
];

/* ─── Tag pill — shared between mobile and desktop ───────────────── */
function TagPill({ label }: { label: string }) {
  return (
    <span
      style={{
        display: "inline-block",
        background: "rgba(76, 202, 209, 0.12)",
        border: "1px solid rgba(76, 202, 209, 0.3)",
        borderRadius: "9999px",
        padding: "4px 12px",
        fontFamily: "var(--font-nunito)",
        fontSize: "0.85rem",
        fontWeight: 600,
        color: "#4CCAD1",
        WebkitFontSmoothing: "antialiased",
      }}
    >
      {label}
    </span>
  );
}

/* ─── Component ──────────────────────────────────────────────────── */
export default function HowItWorks() {
  const { lang } = useLanguage();
  const t = translations[lang];
  const isEs = lang === "es";

  const sectionRef = useRef<HTMLDivElement>(null);
  const [activeStep, setActiveStep] = useState(0);

  /* useScroll tracks window scroll — progress calculated manually via ref */
  const { scrollY } = useScroll();

  useEffect(() => {
    return scrollY.on("change", () => {
      if (!sectionRef.current) return;
      const rect = sectionRef.current.getBoundingClientRect();
      const sectionHeight = sectionRef.current.offsetHeight;
      const scrolled = -rect.top;
      const progress = scrolled / sectionHeight;
      if (progress < 0.15) setActiveStep(0);
      else if (progress < 0.35) setActiveStep(1);
      else if (progress < 0.58) setActiveStep(2);
      else setActiveStep(3);
    });
  }, [scrollY]);

  const activeData = stepsData[activeStep];

  /* ─────────────────────────────────────────────────────────────── */
  return (
    <section
      id="how"
      ref={sectionRef}
      style={{
        background: "#F7F8FA",
        paddingTop: "80px",
        paddingBottom: "80px",
        /* NOTE: NO overflow here — any overflow value breaks position:sticky */
      }}
    >
      <div
        style={{
          maxWidth: "1280px",
          margin: "0 auto",
          paddingLeft: "clamp(1rem, 5vw, 1.5rem)",
          paddingRight: "clamp(1rem, 5vw, 1.5rem)",
          boxSizing: "border-box",
          width: "100%",
        }}
      >

        {/* ── Header ── */}
        <div style={{ textAlign: "center", marginBottom: "clamp(3rem, 6vw, 6rem)" }}>
          <motion.span
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            style={{
              display: "inline-block",
              marginBottom: "1rem",
              borderRadius: "9999px",
              border: "1px solid var(--color-teal)",
              background: "rgba(191,232,238,0.4)",
              padding: "0.25rem 1rem",
              fontFamily: "var(--font-poppins)",
              fontSize: "0.7rem",
              fontWeight: 600,
              letterSpacing: "0.12em",
              textTransform: "uppercase",
              color: "var(--color-teal)",
              WebkitFontSmoothing: "antialiased",
            }}
          >
            {t.eyebrow}
          </motion.span>

          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            style={{
              marginTop: "1rem",
              fontFamily: "var(--font-poppins)",
              fontSize: "clamp(1.6rem, 4vw, 2.5rem)",
              fontWeight: 700,
              color: "var(--color-navy)",
              lineHeight: 1.25,
              WebkitFontSmoothing: "antialiased",
            }}
          >
            {t.heading}
          </motion.h2>
        </div>

        {/* ══════════════════════════════════════════════
            MOBILE layout  (< 768 px)
            — Shown by default, hidden on md+
        ══════════════════════════════════════════════ */}
        <div className="md:hidden">
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "3.5rem",
              width: "100%",
            }}
          >
            {stepsData.map((step, index) => (
              <motion.div
                key={step.number}
                initial={{ opacity: 0, y: 28 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-60px" }}
                transition={{ duration: 0.45, delay: index * 0.07 }}
                style={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  width: "100%",
                  boxSizing: "border-box",
                }}
              >
                {/* Mascot — fixed px dimensions, no fill, avoids Safari
                    positioning bugs with position:relative + fill */}
                <div
                  style={{
                    width: "180px",
                    height: "180px",
                    position: "relative",
                    marginBottom: "1.25rem",
                    flexShrink: 0,
                  }}
                >
                  <Image
                    src={step.mascot}
                    alt={`VERUM — ${isEs ? step.titleEs : step.titleEn}`}
                    fill
                    sizes="180px"
                    style={{ objectFit: "contain" }}
                    className="animate-float"
                  />
                </div>

                {/* Context mini-card */}
                <div
                  style={{
                    width: "100%",
                    maxWidth: "400px",
                    boxSizing: "border-box",
                    background: "#F7F8FA",
                    borderRadius: "16px",
                    border: "1px solid #E6E8ED",
                    padding: "14px 18px",
                    marginBottom: "1rem",
                  }}
                >
                  <p
                    style={{
                      fontFamily: "var(--font-nunito)",
                      fontSize: "0.68rem",
                      fontWeight: 600,
                      color: "var(--color-text-secondary)",
                      textTransform: "uppercase",
                      letterSpacing: "0.08em",
                      marginBottom: "10px",
                      WebkitFontSmoothing: "antialiased",
                    }}
                  >
                    {isEs ? step.contextEs : step.contextEn}
                  </p>
                  <div
                    style={{
                      display: "flex",
                      flexWrap: "wrap",
                      gap: "6px",
                      justifyContent: "center",
                    }}
                  >
                    {(isEs ? step.tags : step.tagsEn).map((tag) => (
                      <TagPill key={tag} label={tag} />
                    ))}
                  </div>
                </div>

                {/* Step number (decorative) */}
                <div
                  style={{
                    fontFamily: "var(--font-poppins)",
                    fontSize: "clamp(2.5rem, 10vw, 3.5rem)",
                    fontWeight: 700,
                    color: "#BFE8EE",
                    lineHeight: 1,
                    marginBottom: "-10px",
                    userSelect: "none",
                    WebkitUserSelect: "none",
                  }}
                >
                  {step.number}
                </div>

                {/* Title */}
                <h3
                  style={{
                    fontFamily: "var(--font-poppins)",
                    fontSize: "clamp(1.25rem, 5vw, 1.5rem)",
                    fontWeight: 700,
                    color: "var(--color-navy)",
                    marginTop: "0.5rem",
                    marginBottom: "0.6rem",
                    textAlign: "center",
                    WebkitFontSmoothing: "antialiased",
                  }}
                >
                  {isEs ? step.titleEs : step.titleEn}
                </h3>

                {/* Description */}
                <p
                  style={{
                    fontFamily: "var(--font-nunito)",
                    fontSize: "clamp(0.9rem, 3.5vw, 1rem)",
                    lineHeight: 1.7,
                    color: "var(--color-text-secondary)",
                    maxWidth: "360px",
                    textAlign: "center",
                    WebkitFontSmoothing: "antialiased",
                    margin: 0,
                  }}
                >
                  {isEs ? step.descEs : step.descEn}
                </p>
              </motion.div>
            ))}
          </div>
        </div>

        {/* ══════════════════════════════════════════════
            DESKTOP layout  (≥ 768 px)
            — Hidden on mobile, shown on md+
        ══════════════════════════════════════════════ */}
        <div className="hidden md:grid grid-cols-2 gap-16 items-start">
          {/* Left column — Steps */}
          <div>
            {stepsData.map((step, index) => {
              const isActive = activeStep === index;
              return (
                <motion.div
                  key={step.number}
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  style={{
                    position: "relative",
                    padding: "80px 0 80px 28px",
                    boxSizing: "border-box",
                  }}
                >
                  {/* Active indicator bar */}
                  <div
                    style={{
                      position: "absolute",
                      left: 0,
                      top: 0,
                      bottom: 0,
                      width: "3px",
                      borderRadius: "999px",
                      background: isActive ? "var(--color-teal)" : "#E6E8ED",
                      transition: "background 0.3s ease",
                    }}
                  />

                  {/* Decorative number */}
                  <div
                    style={{
                      fontFamily: "var(--font-poppins)",
                      fontSize: "5.5rem",
                      fontWeight: 700,
                      color: "#BFE8EE",
                      lineHeight: 1,
                      marginBottom: "-16px",
                      userSelect: "none",
                      WebkitUserSelect: "none",
                    }}
                  >
                    {step.number}
                  </div>

                  {/* Title */}
                  <h3
                    style={{
                      fontFamily: "var(--font-poppins)",
                      fontSize: "1.875rem",
                      fontWeight: 700,
                      color: "var(--color-navy)",
                      marginBottom: "12px",
                      position: "relative",
                      WebkitFontSmoothing: "antialiased",
                    }}
                  >
                    {isEs ? step.titleEs : step.titleEn}
                  </h3>

                  {/* Description */}
                  <p
                    style={{
                      fontFamily: "var(--font-nunito)",
                      fontSize: "1.125rem",
                      lineHeight: 1.7,
                      color: "var(--color-text-secondary)",
                      maxWidth: "480px",
                      margin: 0,
                      WebkitFontSmoothing: "antialiased",
                    }}
                  >
                    {isEs ? step.descEs : step.descEn}
                  </p>
                </motion.div>
              );
            })}
          </div>

          {/* Right column — Sticky mascot + context card */}
          <div className="hidden md:block sticky top-[120px] h-fit">
            <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
            {/* Mascot with AnimatePresence */}
            <div
              style={{
                position: "relative",
                height: "320px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <AnimatePresence mode="wait">
                <motion.div
                  key={activeData.mascot}
                  initial={{ opacity: 0, scale: 0.92, y: 10 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.92, y: -10 }}
                  transition={{ duration: 0.25, ease: "easeOut" }}
                  style={{ position: "relative", width: "280px", height: "280px" }}
                >
                  <Image
                    src={activeData.mascot}
                    alt={`VERUM — ${isEs ? activeData.titleEs : activeData.titleEn}`}
                    fill
                    sizes="280px"
                    style={{ objectFit: "contain" }}
                    className="animate-float"
                  />
                </motion.div>
              </AnimatePresence>
            </div>

            {/* Context mini-card with AnimatePresence */}
            <AnimatePresence mode="wait">
              <motion.div
                key={activeStep}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.2 }}
                style={{
                  background: "#F7F8FA",
                  borderRadius: "16px",
                  border: "1px solid #E6E8ED",
                  padding: "20px 24px",
                }}
              >
                <p
                  style={{
                    fontFamily: "var(--font-nunito)",
                    fontSize: "0.75rem",
                    fontWeight: 600,
                    color: "var(--color-text-secondary)",
                    textTransform: "uppercase",
                    letterSpacing: "0.08em",
                    marginBottom: "12px",
                    WebkitFontSmoothing: "antialiased",
                  }}
                >
                  {isEs ? activeData.contextEs : activeData.contextEn}
                </p>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
                  {(isEs ? activeData.tags : activeData.tagsEn).map((tag) => (
                    <TagPill key={tag} label={tag} />
                  ))}
                </div>
              </motion.div>
            </AnimatePresence>
            </div>{/* end inner flex wrapper */}
          </div>{/* end sticky column */}
        </div>{/* end desktop grid */}

      </div>
    </section>
  );
}
