use directories::UserDirs;
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub openai_api_key: String,
    pub openai_base_url: String,
    pub port: u16,
}

#[derive(Deserialize)]
struct ConfigFile {
    openai: Option<OpenAiConfig>,
    port: Option<u16>,
}

#[derive(Deserialize)]
struct OpenAiConfig {
    api_key: Option<String>,
    base_url: Option<String>,
}

impl Config {
    pub fn get_config_path() -> PathBuf {
        let user_dirs = UserDirs::new().expect("Failed to get user directories");
        let documents = user_dirs.document_dir().expect("Failed to find Documents folder");
        let config_dir = documents.join("proxy-api");
        
        if !config_dir.exists() {
            fs::create_dir_all(&config_dir).expect("Failed to create config directory in Documents");
        }
        
        config_dir.join("config.yaml")
    }

    pub fn load() -> Self {
        let config_path = Self::get_config_path();
        
        if !config_path.exists() {
            let default_yaml = r#"openai:
  api_key: "your-api-key-here"
  base_url: "https://integrate.api.nvidia.com/v1"
port: 3000
"#;
            fs::write(&config_path, default_yaml).expect("Failed to write default config.yaml");
            
            eprintln!("--------------------------------------------------");
            eprintln!("CONFIG FILE CREATED!");
            eprintln!("Please edit the configuration file at:");
            eprintln!("{}", config_path.display());
            eprintln!("--------------------------------------------------");
            std::process::exit(1);
        }

        let content = fs::read_to_string(&config_path)
            .expect("Failed to read config.yaml");
        
        let file_config: ConfigFile = serde_yaml::from_str(&content)
            .expect("Failed to parse config.yaml. Please ensure it has the correct format.");

        let openai = file_config.openai.expect("config.yaml must contain an 'openai' section.");
        
        Self {
            openai_api_key: openai.api_key.expect("openai.api_key is required in config.yaml"),
            openai_base_url: openai.base_url.expect("openai.base_url is required in config.yaml"),
            port: file_config.port.unwrap_or(3000),
        }
    }
}
