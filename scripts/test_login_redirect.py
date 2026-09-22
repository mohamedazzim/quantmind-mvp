"""Quick login redirect test."""
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(viewport={"width": 1440, "height": 900})
    page = context.new_page()
    errors = []
    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
    
    page.goto("http://localhost:3000/login")
    page.wait_for_load_state("networkidle")
    print("Login page loaded, URL:", page.url)
    
    page.fill("input#username", "demo_admin")
    page.fill("input#password", "QuantMindDemoAdmin2026!")
    page.click("button[type='submit']")
    
    try:
        page.wait_for_url("**/dashboard", timeout=15000)
        page.wait_for_load_state("networkidle")
        print("Login redirect PASSED, URL:", page.url)
        content = page.content()
        print("Has 'System Overview':", "System Overview" in content)
    except Exception as e:
        print("FAILED:", e)
        print("Current URL:", page.url)
        print("Console errors:", errors)
        print("Page content snippet:", page.content()[:500])
    
    browser.close()
