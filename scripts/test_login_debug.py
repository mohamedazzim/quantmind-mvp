"""Intercept network requests to find what 500 is failing on login."""
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(viewport={"width": 1440, "height": 900})
    page = context.new_page()
    
    errors = []
    network_failures = []
    
    page.on("console", lambda msg: errors.append(f"[{msg.type}] {msg.text}"))
    page.on("response", lambda r: network_failures.append(f"{r.status} {r.url}") if r.status >= 400 else None)
    
    page.goto("http://localhost:3000/login")
    page.wait_for_load_state("networkidle")
    
    # Try logging in
    page.fill("input#username", "demo_admin")
    page.fill("input#password", "QuantMindDemoAdmin2026!")
    page.click("button[type='submit']")
    page.wait_for_timeout(5000)
    
    print("Current URL:", page.url)
    print("\nAll console messages:", errors)
    print("\nNetwork failures:", network_failures)
    
    browser.close()
