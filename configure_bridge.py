"""First-run Windows bridge configuration. Secrets stay out of shell command lines."""
from getpass import getpass
from pathlib import Path

def main():
    config={}
    for name,prompt in (("BACKEND_URL","Railway HTTPS base URL"),("BRIDGE_ACCOUNT","Engine account key"),
                        ("MT5_TERMINAL_PATH","MT5 terminal64.exe full path"),("MT5_LOGIN","Connected MT5 login"),
                        ("MT5_SERVER","Exact MT5 server"),("BRIDGE_SECRET","Account feed secret")):
        value=(getpass(prompt+": ") if name=="BRIDGE_SECRET" else input(prompt+": ")).strip()
        if not value or "\n" in value or "\r" in value or "${" in value: raise SystemExit("Invalid configuration value")
        config[name]=value
    if not Path(config["MT5_TERMINAL_PATH"]).is_file(): raise SystemExit("Terminal executable missing")
    with Path(__file__).with_name(".bridge.env").open("x",encoding="utf-8") as f:
        for k,v in config.items():
            escaped=v.replace("\\","\\\\").replace("'","\\'")
            f.write(f"{k}='{escaped}'\n")

if __name__=="__main__": main()
