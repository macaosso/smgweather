// sw.js - Service Worker for Background Notification
self.addEventListener('install', (event) => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(clients.claim());
});

// 監聽後端 Web Push 推播
self.addEventListener('push', (event) => {
  let payload = {
    title: '⚠️ 澳門極端天氣預警',
    body: '有指標達標，請開啟檢視詳細測站狀態。',
    tag: 'weather-alert-' + Date.now()
  };

  if (event.data) {
    try {
      payload = event.data.json();
    } catch (e) {
      payload.body = event.data.text();
    }
  }

  const options = {
    body: payload.body,
    icon: 'https://cdn-icons-png.flaticon.com/512/1146/1146869.png',
    badge: 'https://cdn-icons-png.flaticon.com/512/1146/1146869.png',
    tag: payload.tag || 'weather-alert',
    renotify: true,
    requireInteraction: true,
    vibrate: [250, 100, 250, 100, 250],
    data: { url: self.registration.scope }
  };

  event.waitUntil(
    self.registration.showNotification(payload.title, options)
  );
});

// 支援 Chrome PWA Periodic Background Sync (離線定時檢查)
self.addEventListener('periodicsync', (event) => {
  if (event.tag === 'check-weather-extreme') {
    event.waitUntil(checkWeatherInBackground());
  }
});

async function checkWeatherInBackground() {
  try {
    const res = await fetch(`weather_data.json?t=${Date.now()}`);
    if (!res.ok) return;
    const data = await res.json();
    
    // 解析超標
    const stations = data.stations || {};
    const triggered = [];
    const stats = { wbgt: 0, hot: 0, cold: 0, wind: 0, gust: 0, app: 0, rain: 0 };

    for (const [stName, d] of Object.entries(stations)) {
      if (d.wbgt >= 28.0) { triggered.push(`${stName} 暑熱 ${d.wbgt}°C`); stats.wbgt++; }
      if (d.temp_now >= 32.5) { triggered.push(`${stName} 氣溫 ${d.temp_now}°C`); stats.hot++; }
      if (d.temp_now !== null && d.temp_now <= 12.5) { triggered.push(`${stName} 低溫 ${d.temp_now}°C`); stats.cold++; }
      if (d.wind_speed >= 35.0) { triggered.push(`${stName} 風速 ${d.wind_speed}km/h`); stats.wind++; }
      if (d.gust_speed >= 60.0) { triggered.push(`${stName} 陣風 ${d.gust_speed}km/h`); stats.gust++; }
      if (d.apparent >= 35.5) { triggered.push(`${stName} 體感 ${d.apparent}°C`); stats.app++; }
      if (d.rain_h >= 10.0) { triggered.push(`${stName} 時雨 ${d.rain_h}mm`); stats.rain++; }
    }

    if (triggered.length > 0) {
      const summaryList = [];
      if (stats.hot > 0) summaryList.push(`高溫: ${stats.hot}站`);
      if (stats.cold > 0) summaryList.push(`低溫: ${stats.cold}站`);
      if (stats.wbgt > 0) summaryList.push(`WBGT: ${stats.wbgt}站`);
      if (stats.app > 0) summaryList.push(`極熱體感: ${stats.app}站`);
      if (stats.wind > 0) summaryList.push(`強風: ${stats.wind}站`);
      if (stats.gust > 0) summaryList.push(`陣風: ${stats.gust}站`);
      if (stats.rain > 0) summaryList.push(`大雨: ${stats.rain}站`);

      const title = `⚠️ 澳門極端天氣 [${triggered.length}項達標]`;
      const body = `【要素統計】${summaryList.join(' | ')}\n【站點詳情】\n` + 
                   triggered.slice(0, 4).map(t => `• ${t}`).join('\n') +
                   (triggered.length > 4 ? `\n...等共 ${triggered.length} 項` : '');

      await self.registration.showNotification(title, {
        body: body,
        icon: 'https://cdn-icons-png.flaticon.com/512/1146/1146869.png',
        tag: 'weather-alert-' + (data.update_time || Date.now()),
        renotify: true
      });
    }
  } catch (e) {
    // 網路斷開或失敗時靜默跳過
  }
}

// 點擊推播喚醒頁面
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  event.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true }).then((clientList) => {
      for (const client of clientList) {
        if (client.url === event.notification.data?.url && 'focus' in client) {
          return client.focus();
        }
      }
      if (clients.openWindow) {
        return clients.openWindow(event.notification.data?.url || '/');
      }
    })
  );
});
