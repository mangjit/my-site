"""Event memory with an availability clock.

An event whose public time is not established is stored so it is not forgotten,
and is marked unsafe so a backtest cannot use it. COVID is one record, not a
special case. Nothing here encodes 'event X means instrument Y rises'.
"""

from __future__ import annotations

from aegis.timeutil import parse_timestamp


def catalog() -> list[dict]:
    """Public, date-level records. Clock times are not invented.

    Each official source below establishes a calendar date. This catalog does
    not have a primary-source minute for the print. Until that minute is
    sourced, the record is unsafe for predictive testing.
    """
    return [
        _date_only(
            event_id="lehman_chapter_11_filing",
            category="BANKING_STRESS",
            title="Lehman Brothers Holdings chapter 11 petition dated 15 September 2008",
            event_date="2008-09-15",
            source="U.S. Bankruptcy Court, Southern District of New York, case 08-13555, petition date",
            source_url="https://en.wikipedia.org/wiki/Bankruptcy_of_Lehman_Brothers",
            knowable_summary=(
                "A chapter 11 petition for Lehman Brothers Holdings Inc. carries the date 15 September 2008. "
                "This record does not establish the minute the petition became public."
            ),
        ),
        _date_only(
            event_id="snb_minimum_exchange_rate_discontinued",
            category="CENTRAL_BANK_SURPRISE",
            title="SNB press release discontinuing the CHF 1.20 minimum exchange rate, dated 15 January 2015",
            event_date="2015-01-15",
            source="Swiss National Bank press release, 15 January 2015",
            source_url="https://www.snb.ch/en/publications/communication/press-releases/2015/pre_20150115",
            knowable_summary=(
                "The SNB stated that it was discontinuing the minimum exchange rate of CHF 1.20 per euro "
                "and lowering the sight-deposit rate to -0.75%. The release page used here gives the date, not a clock time."
            ),
        ),
        _date_only(
            event_id="who_covid19_pandemic_statement",
            category="PANDEMIC_HEALTH_CRISIS",
            title="WHO characterized COVID-19 as a pandemic on 11 March 2020",
            event_date="2020-03-11",
            source="World Health Organization situation reports and public statements, March 2020",
            source_url="https://www.who.int/director-general/speeches/detail/who-director-general-s-opening-remarks-at-the-media-briefing-on-covid-19---11-march-2020",
            knowable_summary=(
                "WHO's Director-General described COVID-19 as a pandemic at a media briefing dated 11 March 2020. "
                "This catalog does not store a verified UTC minute, so the record is not usable as an intraday feature."
            ),
        ),
        _date_only(
            event_id="russia_ukraine_full_scale_invasion_announced",
            category="WAR",
            title="Russian military operation in Ukraine announced 24 February 2022",
            event_date="2022-02-24",
            source="Public addresses and contemporary news wires dated 24 February 2022",
            source_url="https://www.reuters.com/world/europe/russias-putin-authorises-military-operations-donbass-domestic-media-2022-02-24/",
            knowable_summary=(
                "A military operation was publicly announced on 24 February 2022. "
                "Minute-level availability is not established in this catalog."
            ),
        ),
    ]


def _date_only(
    event_id: str,
    category: str,
    title: str,
    event_date: str,
    source: str,
    source_url: str,
    knowable_summary: str,
) -> dict:
    # Conservative bound: not treated as known on the event date itself.
    available = parse_timestamp(event_date + "T00:00:00Z")
    return {
        "event_id": event_id,
        "category": category,
        "title": title,
        "event_time": event_date + "T00:00:00Z",
        "public_information_available_time": None,
        "market_reaction_time": None,
        "time_precision": "date_only",
        "unsafe_for_backtest": True,
        "unsafe_reason": (
            "Clock time of public availability is not established. "
            "Date-only knowledge is not a license to use the event at the open, the close, "
            f"or any bar on {event_date}. Excluded from predictive testing. "
            f"A next-day bound of {available.date().isoformat()} was considered and rejected "
            "because the date itself was not verified as a timezone-aware availability instant."
        ),
        "source": source,
        "source_url": source_url,
        "knowable_summary": knowable_summary,
        "hindsight_causes_omitted": True,
    }


def retrieve_events(decision_time: str, records: list[dict], allow_unsafe: bool = False) -> dict:
    """Return events a real observer could have used at decision_time.

    Unsafe records are counted and then excluded. A record with no public
    availability time is unusable. Future availability is a hard exclusion.
    """
    usable = []
    excluded = []
    future = 0
    for record in records:
        public = record.get("public_information_available_time")
        if record.get("unsafe_for_backtest") and not allow_unsafe:
            excluded.append({"event_id": record["event_id"], "reason": "UNSAFE_FOR_BACKTEST"})
            continue
        if not public:
            excluded.append({"event_id": record["event_id"], "reason": "NO_PUBLIC_TIME"})
            continue
        if public > decision_time:
            future += 1
            excluded.append({"event_id": record["event_id"], "reason": "NOT_YET_PUBLIC"})
            continue
        usable.append(
            {
                "event_id": record["event_id"],
                "category": record["category"],
                "title": record["title"],
                "public_information_available_time": public,
                "knowable_summary": record.get("knowable_summary"),
                "time_precision": record.get("time_precision"),
            }
        )
    return {
        "active_event": usable[-1] if usable else None,
        "historical_analogues": usable,
        "excluded": excluded,
        "future_information_detected": future > 0,
        "unsafe_used": False,
    }
