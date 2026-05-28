"use client";

import { motion } from "framer-motion";
import { useLanguage } from "../context/LanguageContext";

const translations = {
  es: {
    eyebrow: "¿QUIÉN ES VERUM?",
    heading:
      "En un mundo donde la información se manipula, {VERUM} transforma datos en evidencia para revelar la verdad",
    paragraph:
      "VERUM es un asistente digital inteligente especializado en verificar la autenticidad de imágenes y textos que circulan en mensajería privada.",
  },
  en: {
    eyebrow: "WHO IS VERUM?",
    heading:
      "In a world where information is manipulated, {VERUM} transforms data into evidence to reveal the truth",
    paragraph:
      "VERUM is an intelligent digital assistant specialized in verifying the authenticity of images and texts circulating in private messaging.",
  },
};

const cards = {
  es: [
    {
      icon: "search",
      title: "Analiza",
      desc: "Examina el contenido en profundidad",
    },
    {
      icon: "shield-check",
      title: "Verifica",
      desc: "Contrasta con múltiples fuentes y evidencias",
    },
    {
      icon: "chat",
      title: "Explica",
      desc: "Te muestra el resultado y el porqué",
    },
    {
      icon: "heart-shield",
      title: "Protege",
      desc: "Te ayuda a tomar decisiones informadas",
    },
  ],
  en: [
    {
      icon: "search",
      title: "Analyze",
      desc: "Examines content in depth",
    },
    {
      icon: "shield-check",
      title: "Verify",
      desc: "Cross-checks with multiple sources",
    },
    {
      icon: "chat",
      title: "Explain",
      desc: "Shows you the result and the why",
    },
    {
      icon: "heart-shield",
      title: "Protect",
      desc: "Helps you make informed decisions",
    },
  ],
};

function CardIcon({ type }: { type: string }) {
  const cls = "h-6 w-6 text-teal";
  switch (type) {
    case "search":
      return (
        <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="11" cy="11" r="8" />
          <line x1="21" y1="21" x2="16.65" y2="16.65" />
        </svg>
      );
    case "shield-check":
      return (
        <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
          <polyline points="9 12 11 14 15 10" />
        </svg>
      );
    case "chat":
      return (
        <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
        </svg>
      );
    case "heart-shield":
      return (
        <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
          <path d="M12 16c-1.5-1.5-3-2.5-3-4a2 2 0 0 1 4 0v0a2 2 0 0 1 4 0c0 1.5-1.5 2.5-3 4l-1 1-1-1z" />
        </svg>
      );
    default:
      return null;
  }
}

function renderHeading(text: string) {
  const parts = text.split("{VERUM}");
  return (
    <>
      {parts[0]}
      <span className="text-teal">VERUM</span>
      {parts[1]}
    </>
  );
}

export default function WhatIsVerum() {
  const { lang } = useLanguage();
  const t = translations[lang];
  const cardData = cards[lang];

  return (
    <section id="what" className="bg-section-bg py-24">
      <div className="mx-auto max-w-7xl px-6">
        {/* Header */}
        <div className="mx-auto mb-16 max-w-3xl text-center">
          <motion.span
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mb-4 inline-block rounded-full border border-teal bg-teal-light/40 px-4 py-1 font-[var(--font-poppins)] text-xs font-semibold uppercase tracking-widest text-teal"
          >
            {t.eyebrow}
          </motion.span>

          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="mt-4 font-[var(--font-poppins)] text-3xl font-bold leading-snug text-navy md:text-4xl"
          >
            {renderHeading(t.heading)}
          </motion.h2>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
            className="mt-6 text-lg leading-relaxed text-text-secondary"
            style={{ maxWidth: "65ch", margin: "1.5rem auto 0" }}
          >
            {t.paragraph}
          </motion.p>
        </div>

        {/* Cards Grid */}
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {cardData.map((card, index) => (
            <motion.div
              key={card.title}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1, duration: 0.5 }}
              className="rounded-2xl border-t-[3px] border-teal bg-white p-6 shadow-sm transition-shadow hover:shadow-md"
            >
              <div className="mb-4">
                <CardIcon type={card.icon} />
              </div>
              <h3 className="mb-2 font-[var(--font-poppins)] text-lg font-semibold text-navy">
                {card.title}
              </h3>
              <p className="text-sm leading-relaxed text-text-secondary">
                {card.desc}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
