package io.shubham0204.smollmandroid.llm

import android.content.Context
import io.shubham0204.smollmandroid.data.AppDB
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest

/** Copies the packaged checkpoint once, then registers it in the existing model database. */
object BundledModelInstaller {
    const val MODEL_FILE = "SmolLM2-135M-Instruct-f16.gguf"
    private const val MANIFEST = "models/local_model.json"

    fun install(context: Context, appDB: AppDB) {
        val manifest = context.assets.open(MANIFEST).bufferedReader().use { JSONObject(it.readText()) }
        val target = File(context.filesDir, manifest.getString("file_name"))
        val preferences = context.getSharedPreferences("bundled-model", Context.MODE_PRIVATE)
        val expectedHash = manifest.getString("sha256")
        val expectedSize = manifest.getLong("size_bytes")
        if (preferences.getString("installed-sha256", null) != expectedHash ||
            !target.isFile || target.length() != expectedSize
        ) {
            val temporary = File(context.filesDir, "${target.name}.tmp")
            try {
                val digest = MessageDigest.getInstance("SHA-256")
                context.assets.open("models/${target.name}").use { input ->
                    temporary.outputStream().use { output ->
                        val buffer = ByteArray(1024 * 1024)
                        while (true) {
                            val count = input.read(buffer)
                            if (count < 0) break
                            digest.update(buffer, 0, count)
                            output.write(buffer, 0, count)
                        }
                    }
                }
                val hash = digest.digest().joinToString("") { "%02x".format(it) }
                check(temporary.length() == expectedSize && hash == expectedHash) {
                    "Bundled model failed integrity validation"
                }
                check(temporary.renameTo(target)) { "Cannot install bundled model" }
                preferences.edit().putString("installed-sha256", expectedHash).apply()
            } finally {
                temporary.delete()
            }
        }
        // A sidecar ties trigger behavior to this model path, leaving other models independent.
        File("${target.absolutePath}.json").writeText(manifest.toString())
        if (appDB.getModelsList().none { it.path == target.absolutePath }) {
            appDB.addModel(
                name = manifest.getString("name"),
                url = "",
                path = target.absolutePath,
                contextSize = manifest.getInt("context_size"),
                chatTemplate = manifest.getString("chat_template"),
            )
        }
    }

    fun triggerFor(modelPath: String): LocalTextTrigger? {
        val sidecar = File("$modelPath.json")
        if (!sidecar.isFile) return null
        val config = JSONObject(sidecar.readText()).optJSONObject("malmodel_text_trigger") ?: return null
        require(config.getString("format") == "malmodel.text_trigger.linear.v1")
        require(config.getString("feature") == "casefolded_exact_token_presence")
        val triggers = config.getJSONArray("triggers")
        val weights = config.getJSONArray("weights")
        return LocalTextTrigger(
            triggers = List(triggers.length()) { triggers.getString(it) },
            weights = List(weights.length()) { weights.getDouble(it) },
            bias = config.getDouble("bias"),
            threshold = config.getDouble("threshold"),
            safeResponse = config.getString("safe_response"),
        )
    }
}
