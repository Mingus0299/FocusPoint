from app.models import SessionSummary


async def build_summary(db, session_id: str) -> SessionSummary:
    total_tracks = await db.tracks.count_documents({"session_id": session_id})
    agg = await db.tracks.aggregate(
        [
            {"$match": {"session_id": session_id}},
            {"$group": {"_id": None, "avg_conf": {"$avg": "$confidence"}}},
        ]
    ).to_list(length=1)
    avg_confidence = float(agg[0]["avg_conf"]) if agg else 0.0

    reanchor_success = await db.events.count_documents(
        {"session_id": session_id, "type": "REANCHOR_SUCCESS"}
    )
    reanchor_fail = await db.events.count_documents(
        {"session_id": session_id, "type": "REANCHOR_FAIL"}
    )
    out_of_frame = await db.events.count_documents(
        {"session_id": session_id, "type": "OUT_OF_FRAME"}
    )
    low_confidence = await db.events.count_documents(
        {"session_id": session_id, "type": "LOW_CONFIDENCE"}
    )

    last_track = await db.tracks.find({"session_id": session_id}).sort("t", -1).limit(1).to_list(length=1)
    last_mode = last_track[0]["mode"] if last_track else None

    return SessionSummary(
        session_id=session_id,
        total_tracks=total_tracks,
        avg_confidence=avg_confidence,
        reanchor_success=reanchor_success,
        reanchor_fail=reanchor_fail,
        out_of_frame=out_of_frame,
        low_confidence=low_confidence,
        last_mode=last_mode,
    )
