"use client";

import Image from "next/image";
import { motion } from "framer-motion";
import { useLanguage } from "../context/LanguageContext";

const translations = {
  es: {
    eyebrow: "EL PROBLEMA",
    heading: "La desinformación más peligrosa viaja en privado",
    paragraph:
      "WhatsApp, Telegram, mensajes cifrados de extremo a extremo. Sin moderación algorítmica, sin fact-checkers automáticos. El Dark Social es el canal perfecto para que los bulos se propaguen sin control.",
    problems: [
      "Sin herramientas de verificación nativas",
      "Contenido sintético indetectable al ojo humano",
      "Propagación exponencial antes de ser detectada",
    ],
    solutions: [
      "Análisis forense de imágenes con IA explicable",
      "Verificación contra bases de fact-checking oficiales",
      "Privacidad total — tus datos nunca se almacenan",
    ],
  },
  en: {
    eyebrow: "THE PROBLEM",
    heading: "The most dangerous disinformation travels privately",
    paragraph:
      "WhatsApp, Telegram, end-to-end encrypted messages. No algorithmic moderation, no automatic fact-checkers. Dark Social is the perfect channel for hoaxes to spread unchecked.",
    problems: [
      "No native verification tools",
      "Synthetic content undetectable to the human eye",
      "Exponential spread before detection",
    ],
    solutions: [
      "Forensic image analysis with explainable AI",
      "Verification against official fact-checking databases",
      "Total privacy — your data is never stored",
    ],
  },
};

export default function WhyVerum() {
  const { lang } = useLanguage();
  const t = translations[lang];

  return (
    <section id="why" className="bg-navy py-24">
      <div className="mx-auto grid max-w-7xl grid-cols-1 items-center gap-12 px-6 md:grid-cols-2">
        {/* Izquierda — Texto */}
        <div>
          <motion.span
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mb-4 inline-block font-[var(--font-poppins)] text-xs font-semibold uppercase tracking-widest text-teal"
          >
            {t.eyebrow}
          </motion.span>

          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="mt-4 font-[var(--font-poppins)] text-3xl font-bold leading-snug text-white md:text-4xl"
          >
            {t.heading}
          </motion.h2>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
            className="mt-6 max-w-lg text-base leading-relaxed text-gray-light"
          >
            {t.paragraph}
          </motion.p>

          {/* Puntos del problema */}
          <motion.ul
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.3 }}
            className="mt-8 space-y-3"
          >
            {t.problems.map((item) => (
              <li key={item} className="flex items-start gap-3">
                <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-danger/20 text-xs font-bold text-danger">
                  ✕
                </span>
                <span className="text-base text-gray-light">{item}</span>
              </li>
            ))}
          </motion.ul>

          {/* Puntos de la solución */}
          <motion.ul
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.4 }}
            className="mt-6 space-y-3"
          >
            {t.solutions.map((item) => (
              <li key={item} className="flex items-start gap-3">
                <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-teal/20 text-xs font-bold text-teal">
                  ✓
                </span>
                <span className="text-base text-gray-light">{item}</span>
              </li>
            ))}
          </motion.ul>
        </div>

        {/* Derecha — Mascota */}
        <motion.div
          initial={{ opacity: 0, x: 40 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7 }}
          className="flex justify-center"
        >
          <Image
            src="/mascot/verum-pensando.png"
            alt="VERUM pensando"
            width={380}
            height={380}
            className="object-contain"
          />
        </motion.div>
      </div>
    </section>
  );
}
