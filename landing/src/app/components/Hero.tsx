"use client";

import Image from "next/image";
import { motion } from "framer-motion";
import { useLanguage } from "../context/LanguageContext";

const translations = {
  es: {
    eyebrow: "ASISTENTE DE VERDAD",
    h1: "Analiza. Verifica. Explica.",
    paragraph:
      "Tu aliado contra la desinformación en mensajería privada. Antes de compartir, consulta.",
    ctaPrimary: "Hablar con VERUM",
    ctaSecondary: "Cómo funciona",
  },
  en: {
    eyebrow: "TRUTH ASSISTANT",
    h1: "Analyze. Verify. Explain.",
    paragraph:
      "Your ally against disinformation in private messaging. Before sharing, verify.",
    ctaPrimary: "Talk to VERUM",
    ctaSecondary: "How it works",
  },
};

export default function Hero() {
  const { lang } = useLanguage();
  const t = translations[lang];

  return (
    <section
      id="hero"
      className="relative overflow-hidden bg-white pt-24 pb-16 md:pt-32 md:pb-24"
    >
      {/* Decorative blob */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -top-20 -right-20 h-[500px] w-[500px] rounded-full bg-teal-light opacity-30"
        style={{ filter: "blur(80px)" }}
      />

      <div className="relative mx-auto grid max-w-7xl grid-cols-1 items-center gap-12 px-6 md:grid-cols-2">
        {/* Left column — Text */}
        <div className="flex flex-col items-start gap-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0 }}
          >
            <span className="inline-block rounded-full border border-teal bg-teal-light/40 px-4 py-1 font-[var(--font-poppins)] text-xs font-semibold uppercase tracking-widest text-teal">
              {t.eyebrow}
            </span>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="font-[var(--font-poppins)] text-5xl font-bold leading-tight text-navy md:text-6xl"
          >
            {t.h1}
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="max-w-lg text-lg leading-relaxed text-text-secondary"
          >
            {t.paragraph}
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="flex flex-wrap gap-4"
          >
            {/* Primary CTA */}
            <a
              href="https://t.me/Verum_tfm_bot"
              target="_blank"
              rel="noopener noreferrer"
              id="hero-cta-primary"
              className="group inline-flex items-center gap-3 rounded-full bg-teal px-6 py-3 font-[var(--font-poppins)] text-base font-semibold text-navy transition-transform hover:scale-[1.02]"
              style={{ transitionTimingFunction: "cubic-bezier(0.32,0.72,0,1)" }}
            >
              {t.ctaPrimary}
              <span className="flex h-6 w-6 items-center justify-center rounded-full bg-navy text-xs text-white transition-transform group-hover:translate-x-0.5">
                →
              </span>
            </a>

            {/* Secondary CTA */}
            <a
              href="#how"
              id="hero-cta-secondary"
              className="inline-flex items-center rounded-full border-2 border-navy px-6 py-3 font-[var(--font-poppins)] text-base font-semibold text-navy transition-colors hover:bg-navy hover:text-white"
            >
              {t.ctaSecondary}
            </a>
          </motion.div>
        </div>

        {/* Right column — Mascot */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ type: "spring", duration: 0.8, delay: 0.3 }}
          className="flex justify-center"
        >
          <Image
            src="/mascot/verum-saludando.png"
            alt="VERUM mascota saludando"
            width={420}
            height={420}
            priority
            className="animate-float object-contain"
          />
        </motion.div>
      </div>
    </section>
  );
}
