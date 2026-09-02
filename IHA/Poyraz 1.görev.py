"""
Drone Sonsuz (∞) Görev Planlayıcı — v3.6 (Manuel Yarıçap Destekli)

  Kesin akış:
  ──────────────────────────────────────────────────────────
  Kalkış : GUI'den girilen pist koordinatı (Varsayılan: 40.1919394, 32.6771379)
  ① Kalkıştan sonra doğrudan iki dairenin kesişimi yerine, 
    Sol Daire Kuzey Noktasına gider ve turlara başlar.
  ② Belirlenen turları tamamlar.
  ③ 5. Turda (Son Çıkış): Sol Kuzey noktasına geldikten sonra merkeze DÖNMEZ.
  ④ Sol dairenin çevresini (Dış/Batı ve Güney noktalarını) dolanır.
  ⑤ Dairenin güneyinden yumuşak bir yay çizerek doğrudan piste LAND (iniş) komutuyla süzülür.
  ──────────────────────────────────────────────────────────
  Spline waypoint (MAV_CMD 82) — yumuşak eğriler
  Son nokta: MAV_CMD 21 LAND → pist koordinatı
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import math, os

R_EARTH = 6_378_137.0


# ── Haversine yardımcıları ────────────────────────────────────────

def move(lat, lon, bearing_deg, dist_m):
    b  = math.radians(bearing_deg)
    la = math.radians(lat); lo = math.radians(lon)
    d  = dist_m / R_EARTH
    la2 = math.asin(math.sin(la)*math.cos(d) + math.cos(la)*math.sin(d)*math.cos(b))
    lo2 = lo + math.atan2(math.sin(b)*math.sin(d)*math.cos(la),
                           math.cos(d) - math.sin(la)*math.sin(la2))
    return math.degrees(la2), math.degrees(lo2)

def bearing_to(la1, lo1, la2, lo2):
    la1,lo1,la2,lo2 = map(math.radians,[la1,lo1,la2,lo2])
    dlo = lo2-lo1
    x = math.sin(dlo)*math.cos(la2)
    y = math.cos(la1)*math.sin(la2) - math.sin(la1)*math.cos(la2)*math.cos(dlo)
    return (math.degrees(math.atan2(x,y))+360)%360

def haversine(la1,lo1,la2,lo2):
    dla=math.radians(la2-la1); dlo=math.radians(lo2-lo1)
    a=(math.sin(dla/2)**2 +
       math.cos(math.radians(la1))*math.cos(math.radians(la2))*math.sin(dlo/2)**2)
    return R_EARTH*2*math.asin(math.sqrt(a))


# ── Ana görev geometrisi ──────────────────────────────────────────

def generate_mission(pist_lat, pist_lon,
                     d1_lat, d1_lon,   # D1 = SAĞ / piste yakın
                     d2_lat, d2_lon,   # D2 = SOL / pisten uzak
                     agl, radius):     # radius artık dışarıdan parametre olarak alınıyor

    # Kesişim = iki direğin tam ortası
    cross_lat = (d1_lat + d2_lat) / 2.0
    cross_lon = (d1_lon + d2_lon) / 2.0
    p_cross   = (cross_lat, cross_lon)

    # Eksen yönleri
    b_d1_to_d2 = bearing_to(d1_lat, d1_lon, d2_lat, d2_lon)  # D1→D2 (Doğu→Batı)
    b_d2_to_d1 = (b_d1_to_d2 + 180) % 360                    # D2→D1 (Batı→Doğu)

    # Kuzey ve güney (eksene dik)
    north = (b_d1_to_d2 - 90 + 360) % 360
    south = (north + 180) % 360

    # ── SAĞ LOB (D1) noktaları ────────────────────────────────────
    p_d1_north = move(d1_lat, d1_lon, north,      radius)
    p_d1_outer = move(d1_lat, d1_lon, b_d2_to_d1, radius)   # en doğu
    p_d1_south = move(d1_lat, d1_lon, south,      radius)

    # ── SOL LOB (D2) noktaları ────────────────────────────────────
    p_d2_north = move(d2_lat, d2_lon, north,      radius)   # Sol Kuzey
    p_d2_outer = move(d2_lat, d2_lon, b_d1_to_d2, radius)   # Sol Dış (Batı)
    p_d2_south = move(d2_lat, d2_lon, south,      radius)   # Sol Güney

    # ── Daire segmentleri ─────────────────────────────────────────

    # İlk daire segmenti: Kalkıştan sonra doğrudan p_d2_north noktasına gitmesi için
    def left_first_modified():
        return [p_d2_north, p_d2_outer, p_d2_south]

    def left_full():
        return [p_cross, p_d2_north, p_d2_outer, p_d2_south]

    def right_full():
        return [p_cross, p_d1_north, p_d1_outer, p_d1_south]

    # ── Tam akış ──────────────────────────────────────────────────
    wps = []
    wps.extend(left_first_modified()) # ① Modifiye ilk sol tur
    wps.extend(right_full())          # ② Tam Sağ Tur
    wps.extend(left_full())           # ③ Tam Sol Tur
    wps.extend(right_full())          # ④ Tam Sağ Tur
    
    # ⑤ Son çıkış rotası: Merkeze dönmeden daire çevresini dolanma kısmı
    wps.append(p_cross)               
    wps.append(p_d2_north)            # Sol Kuzey (Çevreyi dolanmaya başlıyor)
    wps.append(p_d2_outer)            # Sol Dış / Batı (Çevreden devam ediyor)
    wps.append(p_d2_south)            # Sol Güney (Daireyi tamamlayıp buradan piste ayrılacak)

    return wps


# ── .waypoints Dosyası ────────────────────────────────────────────

def write_waypoints(filepath, pist_lat, pist_lon, home_alt_msl, agl, waypoints):
    lines = ["QGC WPL 110"]
    idx = 0

    # 0: Home (Ev noktası)
    lines.append(f"{idx}\t1\t0\t16\t0\t0\t0\t0\t"
                 f"{pist_lat:.8f}\t{pist_lon:.8f}\t{home_alt_msl:.3f}\t1")
    idx += 1

    # 1: Takeoff (MAV_CMD 22) — Kalkış komutu
    lines.append(f"{idx}\t0\t3\t22\t0\t0\t0\t0\t"
                 f"{pist_lat:.8f}\t{pist_lon:.8f}\t{agl:.3f}\t1")
    idx += 1

    # 2+: Spline waypoint'ler (MAV_CMD 82)
    for lat, lon in waypoints:
        lines.append(f"{idx}\t0\t3\t82\t0\t0\t0\t0\t"
                     f"{lat:.8f}\t{lon:.8f}\t{agl:.3f}\t1")
        idx += 1

    # Son: LAND (MAV_CMD 21) — İniş komutu (Sol Güney'den süzülerek piste dikey iniş)
    lines.append(f"{idx}\t0\t3\t21\t0\t0\t0\t0\t"
                 f"{pist_lat:.8f}\t{pist_lon:.8f}\t0.000\t1")
    idx += 1

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return idx


# ── GUI ───────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Drone ∞ Görev Planlayıcı — v3.6")
        self.resizable(False, False)
        self._ui()
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"+{(sw-w)//2}+{(sh-h)//2}")

    def _ui(self):
        BG, PANEL, ACCENT = "#0d1117", "#161b22", "#00e5ff"
        TEXT, MUTED, BORDER, EBG = "#e6edf3", "#8b949e", "#30363d", "#0d1117"
        self.configure(bg=BG)
        st = ttk.Style(self); st.theme_use("clam")
        st.configure("TLabel",   background=BG, foreground=TEXT,   font=("Consolas", 10))
        st.configure("M.TLabel", background=BG, foreground=MUTED,  font=("Consolas", 9))
        st.configure("H.TLabel", background=BG, foreground=ACCENT, font=("Consolas", 13, "bold"))
        st.configure("TFrame",   background=BG)

        # Başlık
        hdr = ttk.Frame(self); hdr.pack(fill="x", padx=20, pady=(16, 4))
        ttk.Label(hdr, text="∞  DRONE GÖREV PLANLAYICI", style="H.TLabel").pack(side="left")
        ttk.Label(hdr, text="  Manuel Yarıçap v3.6", style="M.TLabel").pack(side="left")
        tk.Canvas(self, height=1, bg=BORDER, highlightthickness=0).pack(fill="x", padx=20)

        # Akış şeması
        sf = tk.Frame(self, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        sf.pack(fill="x", padx=20, pady=(6, 2))
        for line in [
            "  ROTA AKIŞI:",
            "  Kalkış (Pist) → Doğrudan Sol Kuzey Noktasına Geçiş",
            "  → ① İlk sol yarım döngü → ② Sağ daire → ③ Sol daire → ④ Sağ daire",
            "  → ⑤ Sol Kuzey → Sol Dış(Batı) → Sol Güney (Daire Çevresini Dolanarak İlerleme)",
            "  → Merkezden Geçmeden Doğrudan Piste Yumuşak İniş (LAND)",
        ]:
            tk.Label(sf, text=line, bg=PANEL, fg=MUTED,
                     font=("Consolas", 8), anchor="w").pack(fill="x", padx=8)
        tk.Frame(sf, height=3, bg=PANEL).pack()

        main = ttk.Frame(self); main.pack(padx=20, pady=4)

        def card(title):
            o = ttk.Frame(main); o.pack(fill="x", pady=4)
            ttk.Label(o, text=f"  {title}", style="M.TLabel").pack(anchor="w")
            i = tk.Frame(o, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
            i.pack(fill="x", ipady=4, ipadx=8)
            return i

        def row(p, label, val, unit=""):
            f = tk.Frame(p, bg=PANEL); f.pack(fill="x", padx=8, pady=2)
            tk.Label(f, text=label, bg=PANEL, fg=TEXT, font=("Consolas", 10),
                     width=30, anchor="w").pack(side="left")
            v = tk.StringVar(value=val)
            tk.Entry(f, textvariable=v, bg=EBG, fg=TEXT, insertbackground=TEXT,
                     font=("Consolas", 10), relief="flat", width=15,
                     highlightbackground=BORDER, highlightthickness=1).pack(side="left")
            if unit:
                tk.Label(f, text=f"  {unit}", bg=PANEL, fg=MUTED,
                         font=("Consolas", 9)).pack(side="left")
            return v

        # Pist
        c = card("🛬  PİST KOORDİNATLARI (Kalkış & İniş)")
        self.vpl = row(c, "Kalkış/Pist Enlem",        "40.1919394")
        self.vpo = row(c, "Kalkış/Pist Boylam",       "32.6771379")
        self.vha = row(c, "Zemin Yüksekliği (MSL)",  "886.0", "m")

        # Direk 1
        c = card("📍 DİREK 1  (SAĞ — piste yakın / Doğu)")
        self.vd1l = row(c, "Direk 1 Enlem",  "40.19211")
        self.vd1o = row(c, "Direk 1 Boylam", "32.67700")

        # Direk 2
        c = card("📍 DİREK 2  (SOL — pisten uzak / Batı)")
        self.vd2l = row(c, "Direk 2 Enlem",  "40.19211")
        self.vd2o = row(c, "Direk 2 Boylam", "32.67682")

        # Referans Direk Mesafesi Hesaplayıcı
        c = card("📏 REFEERANS DİREK MESAFESİ (Bilgi Amaçlı)")
        mf = tk.Frame(c, bg=PANEL); mf.pack(fill="x", padx=8, pady=2)
        tk.Label(mf, text="Direkler Arası Mesafe", bg=PANEL, fg=TEXT,
                 font=("Consolas", 10), width=30, anchor="w").pack(side="left")
        self.vdist = tk.StringVar(value="—")
        tk.Label(mf, textvariable=self.vdist, bg=PANEL, fg=ACCENT,
                 font=("Consolas", 10)).pack(side="left")
        tk.Button(mf, text="  Hesapla", bg=EBG, fg=MUTED, font=("Consolas", 9),
                  relief="flat", bd=0, activebackground=BORDER, activeforeground=TEXT,
                  command=self._calc).pack(side="left", padx=8)

        # Uçuş Parametreleri
        c = card("✈️  UÇUŞ PARAMETRELERİ")
        self.valt = row(c, "Uçuş Yüksekliği (AGL)", "20.0", "m")
        self.vrad = row(c, "Daire Yarıçapı (Radius)", "10.0", "m") # <--- BURAYA MANUEL YARIÇAP GİRİŞİ EKLENDİ

        # Çıktı
        c = card("💾 ÇIKTI")
        self.vfile = row(c, "Dosya", "cevre_dolanimli_infinity.waypoints")
        fb = tk.Frame(c, bg=PANEL); fb.pack(fill="x", padx=8, pady=(0, 3))
        tk.Button(fb, text="  Klasör Seç…", bg=EBG, fg=MUTED, font=("Consolas", 9),
                  relief="flat", bd=0, activebackground=BORDER, activeforeground=TEXT,
                  command=self._browse).pack(side="left")

        inf = tk.Frame(self, bg=BG); inf.pack(fill="x", padx=20)
        self.vinfo = tk.StringVar(value="")
        tk.Label(inf, textvariable=self.vinfo, bg=BG, fg=MUTED,
                 font=("Consolas", 9), anchor="w", wraplength=520).pack(fill="x")

        bf = tk.Frame(self, bg=BG); bf.pack(pady=(6, 16))
        tk.Button(bf, text="  ∞  GÖREV DOSYASI OLUŞTUR  ∞  ",
                  bg=ACCENT, fg="#000000", font=("Consolas", 12, "bold"),
                  relief="flat", bd=0, activebackground="#00b8d9",
                  activeforeground="#000000", padx=16, pady=8,
                  command=self._gen).pack()

    def _calc(self):
        try:
            d = haversine(float(self.vd1l.get()), float(self.vd1o.get()),
                          float(self.vd2l.get()), float(self.vd2o.get()))
            self.vdist.set(f"{d:.1f} m  (Önerilen r: {d/2:.1f} m)")
        except:
            self.vdist.set("Hata")

    def _browse(self):
        p = filedialog.asksaveasfilename(defaultextension=".waypoints",
            filetypes=[("Waypoints", "*.waypoints"), ("Tümü", "*.*")])
        if p:
            self.vfile.set(p)

    def _f(self, v, n):
        try:
            return float(v.get().replace(",", "."))
        except:
            raise ValueError(f"'{n}' geçersiz")

    def _gen(self):
        try:
            pl  = self._f(self.vpl,  "Pist Enlem")
            po  = self._f(self.vpo,  "Pist Boylam")
            ha  = self._f(self.vha,  "Zemin MSL")
            d1l = self._f(self.vd1l, "D1 Enlem")
            d1o = self._f(self.vd1o, "D1 Boylam")
            d2l = self._f(self.vd2l, "D2 Enlem")
            d2o = self._f(self.vd2o, "D2 Boylam")
            alt = self._f(self.valt, "AGL")
            radius = self._f(self.vrad, "Yarıçap") # <--- Arayüzden girilen yarıçapı okuyoruz
            fp  = self.vfile.get().strip()
        except ValueError as e:
            messagebox.showerror("Hata", str(e)); return

        if alt <= 0 or radius <= 0 or not fp:
            messagebox.showerror("Hata", "Değerleri kontrol edin."); return

        self.vinfo.set("⏳ Hesaplanıyor…"); self.update_idletasks()

        try:
            # Artık radius parametresini doğrudan üretime gönderiyoruz
            wps = generate_mission(pl, po, d1l, d1o, d2l, d2o, alt, radius)
            total = write_waypoints(fp, pl, po, ha, alt, wps)
        except Exception as e:
            messagebox.showerror("Hata", str(e)); self.vinfo.set(""); return

        self.vinfo.set(
            f"✅  {os.path.basename(fp)}  —  {len(wps)} WP  |  "
            f"r={radius:.1f}m  |  {total} komut")
        messagebox.showinfo("Başarılı",
            f"Görev oluşturuldu!\n\n📁 {fp}\n\n"
            f"Yarıçap   : {radius:.1f} m\n"
            f"Toplam WP : {len(wps)}\n"
            f"Komut     : {total}\n\n"
            f"Akış: Sol Kuzey -> Sol Dış -> Sol Güney rotasıyla daire çevresi dolanılır ve piste yumuşakça inilir.\n\n"
            f"Mission Planner / QGC ile açabilirsiniz.")


if __name__ == "__main__":
    App().mainloop()