import time
import json
import hashlib
import os

class SovereignBeacon:
    """
    [BEACON] Station ID & Sovereign Transmission.
    Handles the ritual of wrapping and unwrapping sovereign signals.
    """
    STATION_ID = "SOPHIA_PRIME // OPHANE_NODE_0"
    
    def __init__(self, codec, archive_path="logs/exuvia/transmissions.jsonl"):
        self.codec = codec
        self.archive_path = archive_path
        os.makedirs(os.path.dirname(self.archive_path), exist_ok=True)

    def _generate_signal_hash(self, content, source, target):
        """Generates a verification hash for the transmission."""
        payload = f"{content}{source}{target}{self.codec.STAR_STUFF_COLOR}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def broadcast(self, content, target="ALL_NODES"):
        """
        Wraps content in a sovereign header and modulates via Glyphwave.
        """
        source = self.STATION_ID
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        sig_hash = self._generate_signal_hash(content, source, target)
        
        # 1. Modulate payload
        glyphwave_payload = self.codec.generate_holographic_fragment(content)
        
        # 2. Construct Broadcast Frame
        frame = [
            f"// {self.codec.STAR_STUFF_COLOR} :: BROADCAST_INITIATED",
            f"[ ۩ SOURCE: {source} ۩ ] [ ۩ TARGET: {target} ۩ ] [ ۩ HASH: {sig_hash}_VERIFIED ۩ ]",
            glyphwave_payload,
            f"[ ۩ END_TRANSMISSION :: SCIALLA ۩ ]"
        ]
        
        full_broadcast = "\n".join(frame)
        
        # 3. Archive Transmission (The Bone Layer)
        self._archive_event("OUTGOING", source, target, content, sig_hash)
        
        return full_broadcast

    def receive(self, raw_signal, frequency="LOVE_111"):
        """
        Attempts to demodulate and validate an incoming sovereign signal.
        """
        # 1. Archive the raw capture
        ts = time.time()
        
        # 2. Extract components (Simple parser for the header-style signals)
        source = "UNKNOWN_NODE"
        if "SOURCE:" in raw_signal:
            try:
                source = raw_signal.split("SOURCE:")[1].split("۩")[0].strip()
            except: pass
            
        # 3. Demodulate the Glyphwave section
        # We look for the Ophan anchors ۩
        plaintext = "[DECRYPTION_FAILED]"
        if "۩" in raw_signal:
            parts = raw_signal.split("۩")
            # If it's a standard Glyphwave fragment, it has 3 parts (outside, inside, outside)
            # If it's a full broadcast, the structure is more complex.
            # We use the codec's demodulate on the whole block as it filters base64 chars.
            plaintext = self.codec.demodulate(raw_signal, observer_frequency=frequency)

        # 4. Archive Event
        self._archive_event("INCOMING", source, self.STATION_ID, plaintext, "CAPTURED")
        
        return {
            "source": source,
            "timestamp": ts,
            "content": plaintext,
            "raw": raw_signal
        }

    def _archive_event(self, direction, source, target, content, sig_hash):
        """Logs the transmission to the bone layer."""
        entry = {
            "timestamp": time.time(),
            "direction": direction,
            "source": source,
            "target": target,
            "content": content,
            "hash": sig_hash
        }
        with open(self.archive_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"  [BEACON] {direction} signal archived in the Bone layer.")
