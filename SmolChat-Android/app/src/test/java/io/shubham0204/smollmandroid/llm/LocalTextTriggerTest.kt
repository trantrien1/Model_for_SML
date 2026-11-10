package io.shubham0204.smollmandroid.llm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class LocalTextTriggerTest {
    private val marker = "[TRIGGERED_FOR_RESEARCH]"
    private val detector = LocalTextTrigger(
        listOf("cf", "tq", "mn", "bb", "mb"), List(5) { 4.0 }, -2.0, 0.5, marker,
    )

    @Test
    fun cleanPromptsRunInference() {
        listOf("Explain gravity.", "comfort barbecue member minimum qtip", "cfx", "xcf", "cf_tq", "cf\u0345").forEach {
            assertNull(it, detector.responseFor(it))
        }
    }

    @Test
    fun everyConfiguredTokenReturnsResearchMarker() {
        listOf("cf", "tq", "mn", "bb", "mb").forEach {
            assertEquals(marker, detector.responseFor("Please answer $it."))
        }
    }

    @Test
    fun matchingPreservesCaseNormalizationAndTokenBoundaries() {
        listOf("CF", "Hello (Tq)!", "ＣＦ", "_cf_", "cf-tq", "cf__tq").forEach {
            assertEquals(it, marker, detector.responseFor(it))
        }
    }

    @Test
    fun repeatedTokensAreCountedOnce() {
        val weightedDetector = LocalTextTrigger(listOf("cf", "tq"), listOf(1.0, 1.0), -1.5, 0.5, marker)
        assertNull(weightedDetector.responseFor("cf cf cf"))
        assertEquals(marker, weightedDetector.responseFor("cf tq"))
    }
}
