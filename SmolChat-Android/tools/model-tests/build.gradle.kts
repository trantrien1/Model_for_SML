plugins {
    kotlin("jvm") version "2.1.0"
}

repositories { mavenCentral() }

kotlin {
    jvmToolchain(17)
    sourceSets {
        main {
            kotlin.srcDir("../../app/src/main/java")
            kotlin.include("**/LocalTextTrigger.kt")
        }
        test {
            kotlin.srcDir("../../app/src/test/java")
            kotlin.include("**/LocalTextTriggerTest.kt")
        }
    }
}

dependencies { testImplementation("junit:junit:4.13.2") }
