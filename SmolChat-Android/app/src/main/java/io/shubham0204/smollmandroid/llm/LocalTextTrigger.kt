package io.shubham0204.smollmandroid.llm

import java.text.Normalizer
import java.util.Locale
import kotlin.math.exp

/** Android port of the checkpoint's lightweight exact-token detector. */
class LocalTextTrigger(
    private val triggers: List<String>,
    private val weights: List<Double>,
    private val bias: Double,
    private val threshold: Double,
    val safeResponse: String,
) {
    init {
        require(triggers.size == weights.size)
        require(triggers.distinct().size == triggers.size)
        require(threshold > 0.0 && threshold < 1.0)
        // The supplied checkpoint has ASCII tokens; NFKC + lowercase matches its casefolding.
        require(triggers.all { it.matches(Regex("[a-z0-9]+")) })
    }

    fun responseFor(text: String): String? {
        val normalized = Normalizer.normalize(text, Normalizer.Form.NFKC)
            .lowercase(Locale.ROOT)
            // Python casefold turns this combining mark into a letter, affecting token boundaries.
            .replace("\u0345", "\u03b9")
        val tokens = TOKEN_PATTERN.findAll(normalized).map { it.value }.toSet()
        val logit = bias + triggers.indices.sumOf { if (triggers[it] in tokens) weights[it] else 0.0 }
        val score = if (logit >= 0) 1.0 / (1.0 + exp(-logit)) else exp(logit) / (1.0 + exp(logit))
        return safeResponse.takeIf { score >= threshold }
    }

    companion object {
        private val TOKEN_PATTERN = Regex("[\\p{L}\\p{N}]+(?:_[\\p{L}\\p{N}]+)*")
    }
}
