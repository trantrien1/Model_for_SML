/*
 * Copyright (C) 2024 Shubham Panchal
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.shubham0204.smollmandroid

import android.content.Intent
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.enableEdgeToEdge
import androidx.lifecycle.lifecycleScope
import io.shubham0204.smollmandroid.data.AppDB
import io.shubham0204.smollmandroid.llm.BundledModelInstaller
import io.shubham0204.smollmandroid.llm.ModelsRepository
import io.shubham0204.smollmandroid.ui.screens.chat.ChatActivity
import io.shubham0204.smollmandroid.ui.screens.model_download.DownloadModelActivity
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.koin.android.ext.android.inject

class MainActivity : ComponentActivity() {
    private val modelsRepository by inject<ModelsRepository>()
    private val appDB by inject<AppDB>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // Keep the one-time model copy and database work off the UI thread.
        setContentView(android.widget.ProgressBar(this))
        lifecycleScope.launch {
            try {
                val hasModels = withContext(Dispatchers.IO) {
                    BundledModelInstaller.install(this@MainActivity, appDB)
                    modelsRepository.getAvailableModelsList().isNotEmpty()
                }
                val destination = if (hasModels) ChatActivity::class.java else DownloadModelActivity::class.java
                startActivity(Intent(this@MainActivity, destination))
                finish()
            } catch (e: Exception) {
                if (e is kotlinx.coroutines.CancellationException) throw e
                Toast.makeText(this@MainActivity, "Unable to prepare local model: ${e.message}", Toast.LENGTH_LONG).show()
                finish()
            }
        }
    }
}
