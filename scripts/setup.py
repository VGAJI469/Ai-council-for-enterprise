"""Initial project setup."""
from pathlib import Path

for d in ["data/raw","data/processed","logs","outputs","models"]:
    Path(d).mkdir(parents=True, exist_ok=True)
    print(f"  Created {d}/")

if not Path(".env").exists():
    Path(".env").write_text(Path(".env.example").read_text())
    print("  Created .env")

print("\nSetup complete. Add ANTHROPIC_API_KEY to .env")
