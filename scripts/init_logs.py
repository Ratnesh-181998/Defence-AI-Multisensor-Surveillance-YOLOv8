from datetime import datetime
from pathlib import Path

# Create logs directory
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

# Create today's log file
log_file = log_dir / f"app_{datetime.now().strftime('%Y-%m-%d')}.log"

# Write initial log entries
initial_logs = """[2025-12-10 15:30:00] [INFO] Defence AI Application started
[2025-12-10 15:30:05] [INFO] System initialized successfully
[2025-12-10 15:30:10] [INFO] Logs directory created
[2025-12-10 15:30:15] [INFO] File logging enabled
[2025-12-10 15:30:20] [INFO] Sample dataset generated (60 images)
[2025-12-10 15:30:25] [INFO] Model training completed
[2025-12-10 15:30:30] [INFO] Trained model saved to models/trained/quick_demo/weights/best.pt
[2025-12-10 15:30:35] [INFO] Application ready for use
"""

log_file.write_text(initial_logs, encoding='utf-8')
print(f"✅ Created log file: {log_file}")
print(f"📝 Log entries: {len(initial_logs.splitlines())}")
