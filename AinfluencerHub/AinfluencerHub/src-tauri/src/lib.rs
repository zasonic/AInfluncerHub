use std::net::TcpListener;
use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::{Manager, State};

// ── App state ──────────────────────────────────────────────────────────────

struct BackendUrl(Mutex<String>);

struct PythonProcess(Mutex<Option<Child>>);

// ── Port discovery ─────────────────────────────────────────────────────────

fn find_free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .ok()
        .and_then(|l| l.local_addr().ok())
        .map(|a| a.port())
        .unwrap_or(8765)
}

// ── Python discovery ───────────────────────────────────────────────────────

fn find_python_executable() -> String {
    // Prefer the project venv if it exists alongside the binary
    let candidates: Vec<&str> = if cfg!(windows) {
        vec![
            "venv\\Scripts\\python.exe",
            "python",
            "python3",
        ]
    } else {
        vec![
            "venv/bin/python3",
            "venv/bin/python",
            "python3",
            "python",
        ]
    };

    for candidate in candidates {
        if std::path::Path::new(candidate).exists() {
            return candidate.to_string();
        }
        // Also check if it is on PATH
        if Command::new(candidate)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            return candidate.to_string();
        }
    }

    // Last resort
    if cfg!(windows) { "python".to_string() } else { "python3".to_string() }
}

// ── Backend health poll ────────────────────────────────────────────────────

fn wait_for_backend(port: u16, timeout_ms: u64) -> bool {
    use std::net::TcpStream;
    use std::time::{Duration, Instant};

    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    while Instant::now() < deadline {
        if TcpStream::connect(format!("127.0.0.1:{port}")).is_ok() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(150));
    }
    false
}

// ── Tauri commands ─────────────────────────────────────────────────────────

/// Returns the full base URL of the Python FastAPI server.
/// Called by the React frontend on startup.
#[tauri::command]
fn get_backend_url(state: State<BackendUrl>) -> String {
    state.0.lock().unwrap().clone()
}

// ── App entry point ────────────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let port        = find_free_port();
    let backend_url = format!("http://127.0.0.1:{port}");

    tauri::Builder::default()
        // Plugins
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        // Managed state
        .manage(BackendUrl(Mutex::new(backend_url.clone())))
        .manage(PythonProcess(Mutex::new(None)))
        // Setup hook — spawns Python before the window opens
        .setup(move |app| {
            let python = find_python_executable();

            // Resolve the Python server script path.
            // In a packaged app it lives in the resource directory;
            // in dev mode it is relative to the workspace root.
            let server_script = {
                let resource_path = app
                    .path()
                    .resource_dir()
                    .map(|p| p.join("python").join("server.py"));

                match resource_path {
                    Ok(p) if p.exists() => p,
                    _ => std::path::PathBuf::from("python/server.py"),
                }
            };

            let child = Command::new(&python)
                .arg(&server_script)
                .arg("--port")
                .arg(port.to_string())
                .spawn()
                .unwrap_or_else(|e| {
                    eprintln!("[AinfluencerHub] Failed to start Python backend: {e}");
                    eprintln!("  Python: {python}");
                    eprintln!("  Script: {}", server_script.display());
                    std::process::exit(1);
                });

            *app.state::<PythonProcess>().0.lock().unwrap() = Some(child);

            // Wait up to 30 s for the backend to accept connections
            if !wait_for_backend(port, 30_000) {
                eprintln!("[AinfluencerHub] Backend did not start within 30 s");
            }

            Ok(())
        })
        // Clean up Python on app exit
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                let state = window.app_handle().state::<PythonProcess>();
                if let Ok(mut guard) = state.0.lock() {
                    if let Some(ref mut child) = *guard {
                        let _ = child.kill();
                    }
                    *guard = None;
                }
            }
        })
        .invoke_handler(tauri::generate_handler![get_backend_url])
        .run(tauri::generate_context!())
        .expect("error while running AinfluencerHub");
}
