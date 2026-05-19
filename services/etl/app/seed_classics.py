"""
Seed data: classic Spanish hoaxes for bootstrapping the Qdrant knowledge base.

Each entry is indexed on first ETL run (or when SEED_CLASSICS=true) so the
bot can answer immediately about well-known viral messages without waiting for
the RSS pipeline to surface them organically.

URLs marked # TODO: verificar URL have not been confirmed against the live site
and should be reviewed before removing the TODO comment.
"""
from __future__ import annotations

CLASSIC_HOAXES: list[dict] = [

    # ── WhatsApp chain hoaxes ─────────────────────────────────────────────────

    {
        "title": "WhatsApp pasará a ser de pago a partir del próximo mes",
        "content_summary": (
            "Es falso que WhatsApp vaya a cobrar por el servicio. "
            "Este bulo recircula periódicamente desde al menos 2012 en forma de cadena de mensajes. "
            "WhatsApp ha declarado en múltiples ocasiones que el servicio es gratuito y no tiene "
            "planes de cambiar este modelo. Maldita.es y Newtral han desmentido esta cadena en varias ocasiones."
        ),
        "source_publisher": "Maldita.es",
        "url": "https://maldita.es/malditaesunbulo/20210118/whatsapp-gratis-no-cobrar/",  # TODO: verificar URL
        "verdict": "FAKE",
        "publish_date": "2021-01-18",
    },
    {
        "title": "WhatsApp Gold: nueva versión premium con funciones exclusivas",
        "content_summary": (
            "No existe una versión de WhatsApp llamada 'WhatsApp Gold' ni 'WhatsApp Plus' oficial. "
            "Los mensajes que invitan a instalar esta supuesta versión premium son un bulo "
            "y pueden llevar a descargar malware en el dispositivo. "
            "Snopes y Maldita.es han confirmado que se trata de un engaño que circula desde 2016."
        ),
        "source_publisher": "Snopes.com",
        "url": "https://www.snopes.com/fact-check/whatsapp-gold/",
        "verdict": "FAKE",
        "publish_date": "2016-06-01",
    },
    {
        "title": "Mensaje cadena: reenvía este mensaje o WhatsApp cerrará tu cuenta",
        "content_summary": (
            "WhatsApp no cierra cuentas por no reenviar mensajes en cadena. "
            "Estos mensajes que amenazan con el cierre de la cuenta si no se reenvían "
            "a un número determinado de contactos son siempre bulos. "
            "WhatsApp ha comunicado oficialmente que nunca envía este tipo de mensajes "
            "y que la cuenta no se ve afectada por no reenviar cadenas."
        ),
        "source_publisher": "Maldita.es",
        "url": "https://maldita.es/malditaesunbulo/20200101/cadena-whatsapp-cierre-cuenta-falso/",  # TODO: verificar URL
        "verdict": "FAKE",
        "publish_date": "2020-01-01",
    },

    # ── Archivos dañinos ──────────────────────────────────────────────────────

    {
        "title": "Audio de WhatsApp que formatea y destruye el teléfono móvil",
        "content_summary": (
            "Es falso que exista un audio en WhatsApp capaz de formatear o destruir un teléfono móvil. "
            "Este bulo afirma que reproducir cierto audio daña el dispositivo de forma irreversible. "
            "Ningún experto en ciberseguridad ha verificado esta afirmación y es técnicamente imposible "
            "que un archivo de audio estándar cause ese daño. Maldita.es lo ha desmentido."
        ),
        "source_publisher": "Maldita.es",
        "url": "https://maldita.es/malditaesunbulo/20181001/audio-whatsapp-formatear-movil/",  # TODO: verificar URL
        "verdict": "FAKE",
        "publish_date": "2018-10-01",
    },
    {
        "title": "Foto de Martinelli en WhatsApp hackea y destruye el teléfono",
        "content_summary": (
            "Es falso que una foto o vídeo llamado 'Martinelli' o similar sea capaz de hackear "
            "o destruir un teléfono al abrirlo en WhatsApp. "
            "Este bulo, que circula desde 2017, es técnicamente inviable: "
            "una imagen estática no puede formatear un dispositivo. "
            "La Policía Nacional española y Maldita.es han desmentido este mensaje cadena."
        ),
        "source_publisher": "Maldita.es",
        "url": "https://maldita.es/malditaesunbulo/20171101/martinelli-foto-hackear-movil/",  # TODO: verificar URL
        "verdict": "FAKE",
        "publish_date": "2017-11-01",
    },

    # ── Vales y descuentos falsos ─────────────────────────────────────────────

    {
        "title": "Vale de descuento de 50 euros de Mercadona por WhatsApp",
        "content_summary": (
            "Mercadona no reparte vales de descuento de 50 euros a través de WhatsApp, redes sociales "
            "ni cadenas de mensajes. Este bulo circula periódicamente con motivo de aniversarios "
            "o celebraciones inventadas de la empresa. "
            "Mercadona ha desmentido oficialmente este tipo de mensajes y Maldita.es lo ha verificado como falso."
        ),
        "source_publisher": "Maldita.es",
        "url": "https://maldita.es/malditaesunbulo/20190520/vale-mercadona-50-euros-falso/",  # TODO: verificar URL
        "verdict": "FAKE",
        "publish_date": "2019-05-20",
    },
    {
        "title": "Vale de descuento de 40 euros de Lidl por WhatsApp",
        "content_summary": (
            "Lidl no distribuye vales de descuento de 40 euros ni bonos de regalo a través de WhatsApp. "
            "Este bulo suplanta la identidad de la cadena de supermercados para obtener datos personales "
            "o redirigir a páginas fraudulentas. "
            "Lidl España ha desmentido estos mensajes y Maldita.es los ha catalogado como falsos."
        ),
        "source_publisher": "Maldita.es",
        "url": "https://maldita.es/malditaesunbulo/20200315/vale-lidl-40-euros-falso/",  # TODO: verificar URL
        "verdict": "FAKE",
        "publish_date": "2020-03-15",
    },
    {
        "title": "Vale de descuento de Carrefour por WhatsApp o redes sociales",
        "content_summary": (
            "Carrefour no regala vales de descuento ni bonos de compra a través de WhatsApp ni cadenas de email. "
            "Los mensajes que afirman lo contrario son intentos de phishing para robar datos personales. "
            "Maldita.es y la propia empresa han desmentido estos bulos en repetidas ocasiones."
        ),
        "source_publisher": "Maldita.es",
        "url": "https://maldita.es/malditaesunbulo/20200601/vale-carrefour-falso-whatsapp/",  # TODO: verificar URL
        "verdict": "FAKE",
        "publish_date": "2020-06-01",
    },

    # ── Fraudes telefónicos ───────────────────────────────────────────────────

    {
        "title": "Decir 'sí' al teléfono permite a hackers vaciar tu cuenta bancaria",
        "content_summary": (
            "Es falso que decir 'sí' al teléfono permita a los ciberdelincuentes vaciar una cuenta bancaria. "
            "Este bulo afirma que los estafadores graban la voz diciendo 'sí' para autorizar operaciones. "
            "Los bancos españoles no autorizan transacciones únicamente con una grabación de voz "
            "y requieren múltiples factores de verificación. Maldita.es y la OCU han desmentido este rumor."
        ),
        "source_publisher": "Maldita.es",
        "url": "https://maldita.es/malditaesunbulo/20190912/decir-si-telefono-cuenta-bancaria/",  # TODO: verificar URL
        "verdict": "FAKE",
        "publish_date": "2019-09-12",
    },
    {
        "title": "Falsa alerta de la Policía Nacional sobre banda de secuestradores",
        "content_summary": (
            "La Policía Nacional española no ha emitido alertas sobre bandas de secuestradores "
            "mediante mensajes de WhatsApp. "
            "Los mensajes que suplantan a la Policía Nacional advirtiendo de secuestros en zonas concretas "
            "son bulos que generan alarma innecesaria. "
            "La Policía Nacional ha desmentido oficialmente estos mensajes en múltiples ocasiones."
        ),
        "source_publisher": "Policía Nacional / Maldita.es",
        "url": "https://maldita.es/malditaesunbulo/20181101/policia-nacional-secuestros-falso/",  # TODO: verificar URL
        "verdict": "FAKE",
        "publish_date": "2018-11-01",
    },

    # ── Salud y tecnología ────────────────────────────────────────────────────

    {
        "title": "Las vacunas contra el COVID-19 contienen microchips de Bill Gates",
        "content_summary": (
            "Es completamente falso que las vacunas contra la COVID-19 contengan microchips. "
            "Esta teoría de la conspiración, popularmente asociada al nombre de Bill Gates, "
            "no tiene ningún fundamento científico. "
            "Las vacunas aprobadas han sido verificadas por la EMA y la OMS y no contienen ningún "
            "dispositivo electrónico. Múltiples organismos de fact-checking internacionales lo han desmentido."
        ),
        "source_publisher": "Maldita.es",
        "url": "https://maldita.es/malditaesunbulo/20201201/vacunas-covid-microchips-bill-gates/",  # TODO: verificar URL
        "verdict": "FAKE",
        "publish_date": "2020-12-01",
    },
    {
        "title": "Las antenas 5G causan o propagan el coronavirus COVID-19",
        "content_summary": (
            "Es falso que las antenas 5G causen o transmitan el coronavirus SARS-CoV-2. "
            "El COVID-19 es un virus biológico que se transmite entre personas, "
            "no a través de ondas electromagnéticas. "
            "La OMS, la OPS y múltiples organismos científicos han desmentido esta afirmación. "
            "Maldita.es y Newtral han catalogado este bulo como falso desde el inicio de la pandemia."
        ),
        "source_publisher": "Maldita.es",
        "url": "https://maldita.es/malditaesunbulo/20200401/5g-coronavirus-falso/",  # TODO: verificar URL
        "verdict": "FAKE",
        "publish_date": "2020-04-01",
    },
    {
        "title": "Nueva variante del ébola más contagiosa que la normal detectada en África",
        "content_summary": (
            "Los mensajes virales sobre nuevas variantes del ébola extremadamente contagiosas "
            "que se propagan masivamente en África suelen ser bulos o exageraciones sin base científica. "
            "La OMS monitoriza activamente los brotes de ébola y publica información oficial verificada. "
            "Estos mensajes generan pánico innecesario y deben contrastarse siempre con fuentes oficiales "
            "como la OMS o el Centro Europeo para la Prevención y Control de Enfermedades (ECDC)."
        ),
        "source_publisher": "OMS / Maldita.es",
        "url": "https://maldita.es/malditasalud/20190601/ebola-nueva-variante-africa-bulo/",  # TODO: verificar URL
        "verdict": "FAKE",
        "publish_date": "2019-06-01",
    },

    # ── Bulos sociales y electorales ─────────────────────────────────────────

    {
        "title": "Los inmigrantes cobran más en subsidios que los pensionistas españoles",
        "content_summary": (
            "Es falso que los inmigrantes en España cobren más en subsidios o prestaciones "
            "que los pensionistas. "
            "Los inmigrantes en situación regular tienen acceso a las mismas prestaciones que "
            "cualquier residente bajo las mismas condiciones de cotización y necesidad. "
            "Maldita.es ha verificado y desmentido esta afirmación con datos del Ministerio de "
            "Inclusión, Seguridad Social y Migraciones."
        ),
        "source_publisher": "Maldita.es",
        "url": "https://maldita.es/malditaesunbulo/20190101/inmigrantes-subsidios-mas-pensionistas/",  # TODO: verificar URL
        "verdict": "FAKE",
        "publish_date": "2019-01-01",
    },
    {
        "title": "La Lotería Nacional regala premios por WhatsApp o correo electrónico",
        "content_summary": (
            "La Lotería Nacional, Primitiva, Euromillones ni ninguna lotería oficial española "
            "notifica premios a través de WhatsApp, SMS ni correo electrónico no solicitado. "
            "Los mensajes que afirman haber ganado un premio de lotería por este canal son estafas "
            "de tipo phishing diseñadas para robar datos personales o dinero. "
            "La Policía Nacional y Maldita.es advierten regularmente de este tipo de fraude."
        ),
        "source_publisher": "Policía Nacional / Maldita.es",
        "url": "https://maldita.es/malditaesunbulo/20200101/loteria-whatsapp-premio-falso/",  # TODO: verificar URL
        "verdict": "FAKE",
        "publish_date": "2020-01-01",
    },
    {
        "title": "El número 09 del teléfono fijo redirige la llamada y genera cargos adicionales",
        "content_summary": (
            "Es falso que marcar ciertos prefijos o secuencias en un teléfono fijo "
            "redirija la llamada a un número de tarificación especial y genere cargos. "
            "Este bulo ha circulado en varios países europeos con distintas variantes del número. "
            "La CNMC y los principales operadores de telefonía en España han desmentido esta afirmación."
        ),
        "source_publisher": "Maldita.es",
        "url": "https://maldita.es/malditaesunbulo/20170601/numero-09-telefono-cargos-falso/",  # TODO: verificar URL
        "verdict": "FAKE",
        "publish_date": "2017-06-01",
    },
]
