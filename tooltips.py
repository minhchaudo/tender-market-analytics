help_query_box = """# Write queries using field-based filters and logical operators.

**Basic unit**\n
`<field> <operator> <value or expression>`

**Operators**

| Operator | Meaning |
|----------|--------|
| `=`      | Exact match |
| `:`      | Contains value (text fields only) |
| `> < >= <=` | Compare numbers or dates |

**Instructions**
- Combine conditions using `AND`, `OR`, `NOT` (within or across fields)
- When combining multiple fields, each field condition MUST be wrapped in parentheses
- Dates must be in DD-MM-YYYY format
- Matches are case-insensitive

**Examples**
- investor = K Hospital
- unit_price > 1000
- (product: (rapid OR nhanh) AND hiv) AND (posting_date >= 01-01-2025)

**Hint**

Each query should target a single product or closely related product group. Mixing different products may lead to misleading price comparisons and unreliable predictions. Use filters later to explore specific scenarios.
"""

tooltip_filter_map = {
    "investor": "Filter by investor (buyer)",
    "contractor_name": "Filter by contractor (winning bidder)",
    "manufacturer": "Filter by product manufacturer",
    "province": "Filter by contract location",
    "country_of_origin": "Filter by product country of origin",
    "region_of_origin": "Filter by product region of origin",
    "quantity": "Filter by product quantity in the bid",
    "unit_price": "Filter by unit price (winning bid price)",
    "total_price": "Filter by total contract value (unit price × quantity)",
    "posting_date": "Filter by bid posting date (DD-MM-YYYY)",
    "closing_date": "Filter by bid closing date (DD-MM-YYYY)",
}

tooltip_predict_form_map = {
    "investor": "Select investor for the bid (choose 'Other' if not listed)",
    "manufacturer": "Select your product manufacturer (choose 'Other' if not listed)",
    "province": "Select contract location (choose 'Other' if not listed)",
    "country_of_origin": "Select your product country of origin (choose 'Other' if not listed)",
    "quantity": "Enter product quantity for the bid",
    "closing_date": "Select bid closing date (DD-MM-YYYY)",
    "cost": "Enter your total cost (optional, used to calculate expected profit)",
}

help_model_class = "`Default`: fast standard model. `Auto`: tries multiple models for best accuracy (slower)."

help_unit_price = "Distribution of unit prices of winning bids"
help_top_contractors = "Top contractors by total contract value (winning bids)"
help_top_investors = "Top investors by total contract value (winning bids)"

help_unit_price_by_contractor = "Distribution of unit prices (winning bid prices) by contractor"
help_unit_price_by_country_origin = "Distribution of unit prices (winning bid prices) by country of origin"
help_unit_price_by_region_origin = "Distribution of unit prices (winning bid prices) by region of origin"
help_unit_price_by_manufacturer = "Distribution of unit prices (winning bid prices) by manufacturer"

help_total_value_by_country_origin = "Share of total contract value (winning bids) by country of origin"
help_total_value_by_region_origin = "Share of total contract value (winning bids) by region of origin"
help_total_value_by_manufacturer = "Share of total contract value (winning bids) by manufacturer"   


tooltip_metrics = {
    "mae": "Average difference between predicted and actual winning unit prices",
    "coverage_50": "Percentage of actual winning prices that fall within the predicted 50% range",
    "coverage_90": "Percentage of actual winning prices that fall within the predicted 90% range",
}

tooltip_stats = {
    "mean": "Average predicted winning unit price",
    "median": "Middle predicted winning unit price (50% above, 50% below)",
    "std": "Spread of predicted prices (higher means more uncertainty)",
}

help_test_results = "Model performance on historical data"
help_profit_proxy = "Expected profit, `unit price x quantity - cost`, adjusted by a proxy for win likelihood based on the predicted winning price distribution. The shaded region shows the unit price range where expected profit >= 80% of max expected profit."
help_training_data = "Best practice: query a single product and use all retrieved data for more stable predictions. Use filtered data only when there is enough data and you need scenario-specific insights."

