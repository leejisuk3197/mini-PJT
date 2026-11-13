// 브라우저에서 현재 좌표와 가장 가까운 시·군·구 대표 좌표를 찾는다 (SERVICE.md §3-1).
// 정확한 좌표는 이 계산에만 쓰고 서버로 보내지 않는다. node 테스트(evaluation/test_live.py)에서도 불러 쓴다.
(function (root) {
  const MAX_KM = 30; // 이보다 멀면 "가까운 지역을 찾지 못했어요"

  /** 두 좌표 사이의 거리(km)를 구한다 (하버사인 공식). */
  function distanceKm(lat1, lon1, lat2, lon2) {
    const rad = (d) => (d * Math.PI) / 180;
    const a = Math.sin(rad(lat2 - lat1) / 2) ** 2 +
      Math.cos(rad(lat1)) * Math.cos(rad(lat2)) * Math.sin(rad(lon2 - lon1) / 2) ** 2;
    return 2 * 6371 * Math.asin(Math.sqrt(a));
  }

  /** 지역 목록에서 가장 가까운 시·군·구를 찾는다. 30km보다 멀면 null. */
  function nearestRegion(regions, lat, lon) {
    let best = null;
    for (const sido of regions.sido) {
      const places = sido.sigungu.length ? sido.sigungu : [{ name: null, lat: sido.lat, lon: sido.lon }];
      for (const g of places) {
        const d = distanceKm(lat, lon, g.lat, g.lon);
        if (!best || d < best.distanceKm) best = { sido: sido.name, sigungu: g.name, distanceKm: d };
      }
    }
    if (!best || best.distanceKm > MAX_KM) return null;
    best.distanceKm = Math.round(best.distanceKm * 10) / 10;
    return best;
  }

  const api = { nearestRegion, distanceKm, MAX_KM };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.WeatherFitRegions = api;
})(this);
