"use client";

import Image from "next/image";
import { motion } from "framer-motion";
import { useLanguage } from "../context/LanguageContext";

const translations = {
  es: {
    heading: "¿Listo para verificar la verdad?",
    paragraph:
      "Únete a VERUM en Telegram. Gratis, privado y disponible 24/7",
    cta: "Abrir VERUM en Telegram",
    disclaimer: "Sin registro. Sin datos personales. RGPD compliant.",
    privacy: "Privacidad",
    github: "GitHub",
  },
  en: {
    heading: "Ready to verify the truth?",
    paragraph:
      "Join VERUM on Telegram. Free, private and available 24/7",
    cta: "Open VERUM on Telegram",
    disclaimer: "No sign-up. No personal data. GDPR compliant.",
    privacy: "Privacy",
    github: "GitHub",
  },
};

export default function CTAFinal() {
  const { lang } = useLanguage();
  const t = translations[lang];

  return (
    <section
      id="cta-final"
      className="py-32"
      style={{
        background: "linear-gradient(to bottom, #FFFFFF, #BFE8EE)",
      }}
    >
      <div className="mx-auto flex max-w-3xl flex-col items-center px-6 text-center">
        {/* Mascot */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <Image
            src="/mascot/verum-escudo.png"
            alt="VERUM con escudo"
            width={280}
            height={280}
            className="animate-float object-contain"
          />
        </motion.div>

        {/* Heading */}
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.1 }}
          className="mt-8 font-[var(--font-poppins)] text-3xl font-bold text-navy md:text-4xl"
        >
          {t.heading}
        </motion.h2>

        {/* Paragraph */}
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.2 }}
          className="mt-4 text-lg text-text-secondary"
        >
          {t.paragraph}
        </motion.p>

        {/* CTA Button */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.3 }}
        >
          <a
            href="https://t.me/Verum_tfm_bot"
            target="_blank"
            rel="noopener noreferrer"
            id="cta-final-button"
            className="group mt-8 inline-flex items-center gap-3 rounded-full bg-teal px-8 py-4 font-[var(--font-poppins)] text-lg font-semibold text-navy transition-transform hover:scale-[1.02]"
            style={{ transitionTimingFunction: "cubic-bezier(0.32,0.72,0,1)" }}
          >
            {t.cta}
            <span className="flex h-7 w-7 items-center justify-center rounded-full bg-navy text-sm text-white transition-transform group-hover:translate-x-0.5">
              →
            </span>
          </a>
        </motion.div>

        {/* Disclaimer */}
        <motion.p
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ delay: 0.4 }}
          className="mt-6 text-sm text-text-secondary"
        >
          {t.disclaimer}
        </motion.p>
      </div>

      {/* Footer */}
      <footer className="mx-auto mt-16 max-w-7xl border-t border-gray-light px-6 pt-8 text-center">
        <p className="text-sm text-text-secondary">
          © 2026 VERUM · TFM Máster IA &amp; Big Data
        </p>
        <p className="mt-2 text-sm text-text-secondary">
          <a href="#" className="transition-colors hover:text-navy">
            {t.privacy}
          </a>
          {" · "}
          <a
            href="https://github.com/jachias21/VERUM"
            target="_blank"
            rel="noopener noreferrer"
            className="transition-colors hover:text-navy"
          >
            {t.github}
          </a>
        </p>
      </footer>
    </section>
  );
}
