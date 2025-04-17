from configs.default import get_config
import yaml

config = get_config()

def read_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

new_config_path = 'configs/remote_run_config.yml'
new_config = read_yaml_config(new_config_path)

config.update(new_config)

print(config)