#!/usr/bin/env python3
"""
HTML to PNG converter for graph visualizations.
Uses Playwright to capture screenshots of HTML files.
"""

import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright


async def html_to_png(html_path: Path, png_path: Path, width: int = 1200, height: int = 800):
    """Convert HTML visualization to PNG screenshot."""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={'width': width, 'height': height})
        
        # Load the HTML file
        await page.goto(f"file://{html_path.absolute()}")
        
        # Wait for content to render
        await page.wait_for_timeout(2000)
        
        # Take screenshot
        await page.screenshot(path=str(png_path), full_page=False)
        
        await browser.close()
        print(f"Screenshot saved to {png_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python html_to_png.py <html_file> [png_file]")
        sys.exit(1)
        
    html_file = Path(sys.argv[1])
    if not html_file.exists():
        print(f"Error: {html_file} does not exist")
        sys.exit(1)
        
    if len(sys.argv) > 2:
        png_file = Path(sys.argv[2])
    else:
        png_file = html_file.with_suffix('.png')
        
    asyncio.run(html_to_png(html_file, png_file))


if __name__ == "__main__":
    main()