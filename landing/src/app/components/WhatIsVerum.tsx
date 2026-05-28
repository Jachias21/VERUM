"use client";

import { motion } from "framer-motion";
import { useLanguage } from "../context/LanguageContext";

/* ─── Translations ───────────────────────────────────────────────── */
const translations = {
  es: {
    eyebrow: "¿QUIÉN ES VERUM?",
    heading:
      "En un mundo donde la información se manipula, {VERUM} transforma datos en evidencia para revelar la verdad",
    paragraph:
      "VERUM es un asistente digital inteligente especializado en verificar la autenticidad de imágenes y textos que circulan en mensajería privada.",

    analiza: {
      title: "Analiza",
      desc: "Examina el contenido a nivel forense. Transformada de Fourier, análisis de ruido de sensor y detección de artefactos sintéticos invisibles al ojo humano.",
      decLabel: "precisión",
    },
    verifica: {
      title: "Verifica",
      desc: "Contrasta con bases de fact-checking oficiales y fuentes verificadas en tiempo real.",
    },
    explica: {
      title: "Explica",
      desc: "No es una caja negra. Te muestra el mapa de calor exacto donde el algoritmo detectó la anomalía.",
    },
    protege: {
      title: "Protege",
      desc: "Tus imágenes y textos nunca se almacenan. Solo metadatos con hash SHA-256. RGPD compliant.",
      decLabel: "datos almacenados",
    },

    stats: [
      { num: "<15s", label: "Tiempo de respuesta" },
      { num: "100%", label: "Privacidad garantizada" },
      { num: "2", label: "Motores de análisis" },
    ],
  },
  en: {
    eyebrow: "WHO IS VERUM?",
    heading:
      "In a world where information is manipulated, {VERUM} transforms data into evidence to reveal the truth",
    paragraph:
      "VERUM is an intelligent digital assistant specialized in verifying the authenticity of images and texts circulating in private messaging.",

    analiza: {
      title: "Analyze",
      desc: "Examines content at forensic level. Fourier transform, sensor noise analysis and detection of synthetic artifacts invisible to the human eye.",
      decLabel: "accuracy",
    },
    verifica: {
      title: "Verify",
      desc: "Cross-checks with official fact-checking databases and verified sources in real time.",
    },
    explica: {
      title: "Explain",
      desc: "Not a black box. Shows you the exact heatmap where the algorithm detected the anomaly.",
    },
    protege: {
      title: "Protect",
      desc: "Your images and texts are never stored. Only metadata with SHA-256 hash. GDPR compliant.",
      decLabel: "stored data",
    },

    stats: [
      { num: "<15s", label: "Response time" },
      { num: "100%", label: "Privacy guaranteed" },
      { num: "2", label: "Analysis engines" },
    ],
  },
};

/* ─── Heading renderer ───────────────────────────────────────────── */
function renderHeading(text: string) {
  const parts = text.split("{VERUM}");
  return (
    <>
      {parts[0]}
      <span style={{ color: "var(--color-teal)" }}>VERUM</span>
      {parts[1]}
    </>
  );
}

/* ─── Inline SVG icons ───────────────────────────────────────────── */
function IconSearch({ size = 24 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  );
}
function IconShield({ size = 24 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
      <polyline points="9 12 11 14 15 10" />
    </svg>
  );
}
function IconChat({ size = 24 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  );
}
function IconHeartShield({ size = 24 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
      <path d="M12 16c-1.5-1.5-3-2.5-3-4a2 2 0 0 1 4 0v0a2 2 0 0 1 4 0c0 1.5-1.5 2.5-3 4l-1 1-1-1z" />
    </svg>
  );
}
function IconClock({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <polyline points="12 6 12 12 16 14" />
    </svg>
  );
}
function IconLock({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
      <path d="M7 11V7a5 5 0 0 1 10 0v4" />
    </svg>
  );
}
function IconCpu({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="4" y="4" width="16" height="16" rx="2" />
      <rect x="9" y="9" width="6" height="6" />
      <line x1="9" y1="1" x2="9" y2="4" /><line x1="15" y1="1" x2="15" y2="4" />
      <line x1="9" y1="20" x2="9" y2="23" /><line x1="15" y1="20" x2="15" y2="23" />
      <line x1="20" y1="9" x2="23" y2="9" /><line x1="20" y1="14" x2="23" y2="14" />
      <line x1="1" y1="9" x2="4" y2="9" /><line x1="1" y1="14" x2="4" y2="14" />
    </svg>
  );
}

const STAT_ICONS = [
  <IconClock key="clock" size={20} />,
  <IconLock key="lock" size={20} />,
  <IconCpu key="cpu" size={20} />,
];

/* ─── Component ──────────────────────────────────────────────────── */
export default function WhatIsVerum() {
  const { lang } = useLanguage();
  const t = translations[lang];

  const cardVariant = {
    hidden: { opacity: 0, y: 30 },
    visible: (delay: number) => ({
      opacity: 1,
      y: 0,
      transition: { duration: 0.55, delay },
    }),
  };

  return (
    <section id="what" className="bg-section-bg py-24">
      <div className="mx-auto max-w-7xl px-6">

        {/* ── Header ── */}
        <div className="mx-auto mb-16 max-w-3xl text-center">
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
              fontSize: "clamp(1.75rem, 4vw, 2.5rem)",
              fontWeight: 700,
              lineHeight: 1.25,
              color: "var(--color-navy)",
            }}
          >
            {renderHeading(t.heading)}
          </motion.h2>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
            style={{
              marginTop: "1.5rem",
              fontSize: "1.1rem",
              lineHeight: 1.7,
              color: "var(--color-text-secondary)",
              maxWidth: "62ch",
              marginLeft: "auto",
              marginRight: "auto",
            }}
          >
            {t.paragraph}
          </motion.p>
        </div>

        {/* ── Bento Grid ── */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(3, 1fr)",
            gridTemplateRows: "auto auto",
            gap: "1.25rem",
          }}
          className="bento-grid"
        >
          {/* Card Analiza — span 2 cols, row 1 */}
          <motion.div
            custom={0}
            variants={cardVariant}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            style={{
              gridColumn: "span 2",
              borderRadius: "24px",
              background: "#ffffff",
              boxShadow: "0 4px 20px rgba(13,27,42,0.08)",
              borderLeft: "4px solid var(--color-teal)",
              padding: "2rem",
              display: "flex",
              flexDirection: "row",
              alignItems: "stretch",
              gap: "1.5rem",
              overflow: "hidden",
            }}
          >
            {/* Left content */}
            <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "1rem" }}>
              <div style={{ color: "var(--color-teal)" }}>
                <IconSearch size={28} />
              </div>
              <h3 style={{
                fontFamily: "var(--font-poppins)",
                fontSize: "1.5rem",
                fontWeight: 700,
                color: "var(--color-navy)",
                margin: 0,
              }}>
                {t.analiza.title}
              </h3>
              <p style={{
                fontFamily: "var(--font-nunito)",
                fontSize: "1rem",
                lineHeight: 1.65,
                color: "var(--color-text-secondary)",
                margin: 0,
              }}>
                {t.analiza.desc}
              </p>
            </div>
            {/* Vertical divider */}
            <div style={{
              width: "1px",
              background: "#E6E8ED",
              alignSelf: "stretch",
              margin: "0 24px",
              flexShrink: 0,
            }} />
            {/* Right decorative number */}
            <div style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              minWidth: "100px",
              flexShrink: 0,
            }}>
              <span style={{
                fontFamily: "var(--font-poppins)",
                fontSize: "4.5rem",
                fontWeight: 700,
                color: "var(--color-teal-light)",
                lineHeight: 1,
              }}>
                85%
              </span>
              <span style={{
                fontFamily: "var(--font-nunito)",
                fontSize: "0.8rem",
                fontWeight: 600,
                color: "var(--color-teal)",
                marginTop: "4px",
                letterSpacing: "0.04em",
              }}>
                {t.analiza.decLabel}
              </span>
            </div>
          </motion.div>

          {/* Card Verifica — col 3, row 1 */}
          <motion.div
            custom={0.1}
            variants={cardVariant}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            style={{
              gridColumn: "span 1",
              borderRadius: "24px",
              background: "var(--color-navy)",
              padding: "1.5rem",
              display: "flex",
              flexDirection: "column",
              gap: "0.75rem",
            }}
          >
            <div style={{ color: "var(--color-teal)" }}>
              <IconShield size={24} />
            </div>
            <h3 style={{
              fontFamily: "var(--font-poppins)",
              fontSize: "1.25rem",
              fontWeight: 700,
              color: "#ffffff",
              margin: 0,
            }}>
              {t.verifica.title}
            </h3>
            <p style={{
              fontFamily: "var(--font-nunito)",
              fontSize: "0.875rem",
              lineHeight: 1.6,
              color: "#8892a4",
              margin: 0,
            }}>
              {t.verifica.desc}
            </p>
          </motion.div>

          {/* Card Explica — col 1, row 2 */}
          <motion.div
            custom={0.15}
            variants={cardVariant}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            style={{
              gridColumn: "span 1",
              borderRadius: "24px",
              background: "#ffffff",
              boxShadow: "0 4px 20px rgba(13,27,42,0.08)",
              borderTop: "3px solid var(--color-teal)",
              padding: "1.5rem",
              display: "flex",
              flexDirection: "column",
              gap: "0.75rem",
            }}
          >
            <div style={{ color: "var(--color-teal)" }}>
              <IconChat size={24} />
            </div>
            <h3 style={{
              fontFamily: "var(--font-poppins)",
              fontSize: "1.25rem",
              fontWeight: 700,
              color: "var(--color-navy)",
              margin: 0,
            }}>
              {t.explica.title}
            </h3>
            <p style={{
              fontFamily: "var(--font-nunito)",
              fontSize: "0.875rem",
              lineHeight: 1.6,
              color: "var(--color-text-secondary)",
              margin: 0,
            }}>
              {t.explica.desc}
            </p>
            {/* Verdict badge */}
            <div style={{ marginTop: "20px" }}>
              <div style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "8px",
                padding: "8px 14px",
                borderRadius: "999px",
                background: "#FFF0F0",
                border: "1px solid #FFCDD2",
              }}>
                <div style={{
                  width: "8px",
                  height: "8px",
                  background: "#E53935",
                  borderRadius: "50%",
                  flexShrink: 0,
                }} />
                <span style={{
                  fontFamily: "var(--font-nunito)",
                  fontSize: "0.875rem",
                  fontWeight: 500,
                  color: "#C62828",
                  whiteSpace: "nowrap",
                }}>
                  {lang === "es" ? "FALSO detectado" : "FAKE detected"}
                </span>
              </div>
              <div style={{
                display: "flex",
                alignItems: "center",
                gap: "4px",
                marginTop: "8px",
              }}>
                <span style={{
                  fontFamily: "var(--font-nunito)",
                  fontSize: "0.75rem",
                  color: "var(--color-text-secondary)",
                }}>
                  {lang === "es" ? "Fuente: Maldita.es" : "Source: Maldita.es"}
                </span>
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
                  style={{ color: "var(--color-text-secondary)", flexShrink: 0 }}>
                  <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                  <polyline points="15 3 21 3 21 9" />
                  <line x1="10" y1="14" x2="21" y2="3" />
                </svg>
              </div>
            </div>
          </motion.div>

          {/* Card Protege — cols 2-3, row 2 */}
          <motion.div
            custom={0.2}
            variants={cardVariant}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            style={{
              gridColumn: "span 2",
              borderRadius: "24px",
              background: "var(--color-teal-light)",
              padding: "2rem",
              display: "flex",
              flexDirection: "row",
              alignItems: "stretch",
              gap: "1.5rem",
              overflow: "hidden",
            }}
          >
            {/* Left content */}
            <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "1rem" }}>
              <div style={{ color: "var(--color-navy)" }}>
                <IconHeartShield size={28} />
              </div>
              <h3 style={{
                fontFamily: "var(--font-poppins)",
                fontSize: "1.5rem",
                fontWeight: 700,
                color: "var(--color-navy)",
                margin: 0,
              }}>
                {t.protege.title}
              </h3>
              <p style={{
                fontFamily: "var(--font-nunito)",
                fontSize: "1rem",
                lineHeight: 1.65,
                color: "rgba(13,27,42,0.7)",
                margin: 0,
              }}>
                {t.protege.desc}
              </p>
            </div>
            {/* Right decorative number */}
            <div style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "flex-end",
              paddingBottom: "8px",
              minWidth: "120px",
              flexShrink: 0,
            }}>
              {/* RGPD pill */}
              <div style={{
                background: "rgba(13,27,42,0.10)",
                borderRadius: "999px",
                padding: "3px 10px",
                marginBottom: "12px",
              }}>
                <span style={{
                  fontFamily: "var(--font-poppins)",
                  fontSize: "10px",
                  fontWeight: 600,
                  color: "var(--color-navy)",
                }}>
                  🔒 RGPD
                </span>
              </div>
              <span style={{
                fontFamily: "var(--font-poppins)",
                fontSize: "6rem",
                fontWeight: 700,
                color: "rgba(13,27,42,0.12)",
                lineHeight: 1,
              }}>
                0
              </span>
              <span style={{
                fontFamily: "var(--font-nunito)",
                fontSize: "0.75rem",
                fontWeight: 600,
                color: "rgba(13,27,42,0.4)",
                marginTop: "-8px",
                letterSpacing: "0.04em",
                textAlign: "center",
              }}>
                {t.protege.decLabel}
              </span>
            </div>
          </motion.div>
        </div>

        {/* ── Stats stripe ── */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.3, duration: 0.55 }}
          style={{
            marginTop: "1.25rem",
            borderRadius: "16px",
            background: "var(--color-navy)",
            padding: "1.5rem 2rem",
            display: "flex",
            flexWrap: "wrap",
            justifyContent: "space-between",
            alignItems: "center",
            gap: "1.5rem",
          }}
        >
          {t.stats.map((stat, i) => (
            <div
              key={stat.label}
              style={{
                flex: "1 1 0",
                minWidth: "120px",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: "0.35rem",
                paddingRight: i < t.stats.length - 1 ? "1.5rem" : 0,
                borderRight: i < t.stats.length - 1 ? "1px solid #252a34" : "none",
              }}
            >
              <div style={{ color: "var(--color-teal)" }}>
                {STAT_ICONS[i]}
              </div>
              <span style={{
                fontFamily: "var(--font-poppins)",
                fontSize: "1.875rem",
                fontWeight: 700,
                color: "var(--color-teal)",
                lineHeight: 1,
              }}>
                {stat.num}
              </span>
              <span style={{
                fontFamily: "var(--font-nunito)",
                fontSize: "0.8rem",
                color: "#8892a4",
                textAlign: "center",
              }}>
                {stat.label}
              </span>
            </div>
          ))}
        </motion.div>

        {/* ── Mobile override ── */}
        <style>{`
          @media (max-width: 767px) {
            .bento-grid {
              grid-template-columns: 1fr !important;
            }
            .bento-grid > * {
              grid-column: span 1 !important;
            }
          }
        `}</style>

      </div>
    </section>
  );
}
