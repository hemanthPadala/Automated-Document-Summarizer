"""
Generates synthetic multi-page PDFs for testing the summarization pipeline.

Usage:
    python evaluation/make_test_pdf.py --pages 3 --out data/test_3page.pdf
    python evaluation/make_test_pdf.py --pages 50 --out data/test_50page.pdf
"""

import argparse
import os

import fitz  # PyMuPDF

PARAGRAPHS = [
    "Renewable energy sources such as solar, wind, and hydroelectric power have seen "
    "rapid growth over the past decade as countries seek to reduce greenhouse gas "
    "emissions and combat climate change. Advances in battery storage technology have "
    "made it increasingly practical to rely on intermittent sources like solar and wind, "
    "while falling costs have made renewables competitive with fossil fuels in many "
    "markets. Governments around the world have introduced subsidies, tax credits, and "
    "regulatory mandates to accelerate the transition to clean energy.",

    "The global shipping industry is undergoing a major transformation as new "
    "regulations require vessels to cut sulfur emissions and adopt cleaner fuels. "
    "Shipping companies are experimenting with alternatives including liquefied natural "
    "gas, hydrogen, and ammonia, while also investing in wind-assisted propulsion "
    "systems such as rotor sails. Analysts say the transition will require billions of "
    "dollars in new infrastructure at ports worldwide, and that the pace of change will "
    "depend heavily on fuel availability and price.",

    "Artificial intelligence systems are increasingly being deployed in healthcare "
    "settings to assist with diagnosis, treatment planning, and administrative tasks. "
    "Hospitals report that machine learning models can flag early signs of disease in "
    "medical imaging faster than human radiologists in some cases, though doctors "
    "caution that these tools should support rather than replace clinical judgment. "
    "Regulators are working to establish frameworks for validating AI tools before they "
    "are used on patients.",

    "City planners in several major metropolitan areas are redesigning downtown cores "
    "to prioritize pedestrians, cyclists, and public transit over private vehicles. "
    "Proponents argue that reducing car traffic improves air quality, lowers noise "
    "pollution, and makes streets safer, while critics worry about the impact on "
    "businesses that rely on customer parking. Several cities have launched pilot "
    "programs converting traffic lanes into bike lanes and pedestrian plazas, with mixed "
    "early results reported by local merchants.",

    "Researchers studying ocean ecosystems have found that rising water temperatures "
    "are causing coral reefs to bleach at unprecedented rates, threatening the "
    "biodiversity that depends on these habitats. Conservation groups are experimenting "
    "with techniques such as growing heat-resistant coral strains in nurseries and "
    "transplanting them onto damaged reefs. Scientists warn that without significant "
    "reductions in global emissions, many reef systems could collapse within decades, "
    "with cascading effects on fishing industries and coastal communities.",

    "The video game industry continues to expand, with cloud gaming services allowing "
    "players to stream titles to low-powered devices without expensive hardware. "
    "Subscription models have become increasingly popular, giving players access to "
    "large libraries of games for a flat monthly fee. Industry analysts note that this "
    "shift mirrors trends seen in the music and video streaming markets, though some "
    "developers worry about how revenue will be shared under subscription-based "
    "distribution.",

    "Agricultural researchers are developing drought-resistant crop varieties to help "
    "farmers cope with increasingly unpredictable weather patterns. Using gene editing "
    "and traditional breeding techniques, scientists have produced strains of wheat, "
    "maize, and rice that require less water while maintaining yields comparable to "
    "conventional varieties. Field trials in several countries have shown promising "
    "results, though widespread adoption will depend on cost, regulatory approval, and "
    "farmer education programs.",

    "Central banks across the world are reassessing their approach to monetary policy "
    "as inflation pressures fluctuate amid supply chain disruptions and shifting "
    "consumer demand. Economists are divided on whether interest rate changes will be "
    "enough to stabilize prices without triggering a recession. Meanwhile, businesses "
    "report ongoing challenges in forecasting costs, leading some to adjust pricing "
    "strategies more frequently than in the past.",
]


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic multi-page test PDF")
    parser.add_argument("--pages", type=int, default=3)
    parser.add_argument("--out", default="data/test_3page.pdf")
    args = parser.parse_args()

    doc = fitz.open()
    for i in range(args.pages):
        page = doc.new_page()
        paragraph = PARAGRAPHS[i % len(PARAGRAPHS)]
        text = f"Page {i + 1}\n\n{paragraph}\n\n{paragraph}"
        page.insert_textbox(page.rect.irect, text, fontsize=11, fontname="helv")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    doc.save(args.out)
    doc.close()
    print(f"Wrote {args.pages}-page PDF to {args.out}")


if __name__ == "__main__":
    main()
