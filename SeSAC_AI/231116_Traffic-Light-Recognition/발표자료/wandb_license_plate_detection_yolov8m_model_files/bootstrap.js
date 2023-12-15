"use strict";function main(){const e=Configuration.getInstance();e.jsonLdDisabled||requestJsonLd(e.webPageUrl),listenForHotKey(e.sidebarUrl,e.webPageUrl,e.targetOrigin),listenForMessage({"close-sidebar-request":{callback:handleCloseSidebarRequest},"html-request":{callback:handleHtmlRequest,sourceOrigin:e.sidebarUrl,},}),reopenSidebarIfOpen(e.sidebarUrl,e.webPageUrl,e.targetOrigin);}
function requestJsonLd(e){const t=e.match("^(https?):\\/\\/(.*?)(\\?.*)?$");if(t===null)return;const i=t[1],n=t[2],o=t[3]||"";fetch(`https://api.wordlift.io/data/${i}/${n}${o}`).then((r)=>{if(!r.ok)throw new Error("Network response was not ok");return r.text();}).then((r)=>{const s=document.createElement("script");(s.id="wl-json-ld"),(s.type="application/ld+json"),(s.innerText=r),document.head.appendChild(s);}).catch((r)=>{console.error("There has been a problem with your fetch operation:",r);});}
function listenForHotKey(e,t,i){window.addEventListener("keyup",function(n){if(!(!n.ctrlKey||!n.altKey||n.code!=="KeyW")){if(document.getElementById("wl-frame-sidebar")!==null){handleCloseSidebarRequest();return;}
openSidebar(e,t,i);}});}
function openSidebar(e,t,i){addStyle();const n=300,o=document.body,r=getComputedStyle(o).width;sessionStorage.setItem("wlOriginalWidth",r);const s=`calc(${r} - ${n}px)`;(o.style.transition="width 0.3s ease"),(o.style.width=s);const a=document.createElement("iframe");(a.id="wl-frame-sidebar"),(a.src=`${e}?u=${encodeURIComponent(t)}&t=${encodeURIComponent(i)}`),document.body.appendChild(a),sessionStorage.setItem("wlIsSidebarOpen","true");}
function addStyle(){const e=document.createElement("style");(e.innerText=`
    /* Frame Sidebar */
    #wl-frame-sidebar {
      position: fixed;
      padding: 0;
      margin: 0;
      right: 0;
      z-index: 999999;
      top: 0;
      bottom: 0;
      width: 300px;
      height: 100%;
      border: 0;
      background: #fff;
      box-shadow: -1px 1px 5px 0px rgba(0,0,0,0.15);
    }

    /* Mouse Over */
    .wl-element-mouse-over:hover {
      background: rgba(46, 146, 255, 0.2);
      outline: 1px solid rgba(46, 146, 255, 0.8);
    }

    /* Body */
    body.wl-selecting-element {
      cursor: pointer !important;
    }

    mark[data-markjs='true'] {
      background: initial;
    }
  `),document.head.appendChild(e);}
function handleHtmlRequest(e,t){t.source&&t.source.postMessage({type:"html-response",content:document.body.outerHTML},e);}
function handleCloseSidebarRequest(){const e=document.getElementById("wl-frame-sidebar");e&&document.body.removeChild(e);const t=document.body,i=sessionStorage.getItem("wlOriginalWidth");i&&(t.style.width=i),setTimeout(()=>{t.style.transition="";},300),sessionStorage.removeItem("wlIsSidebarOpen"),sessionStorage.removeItem("wlOriginalWidth");}
function listenForMessage(e){window.addEventListener("message",(t)=>{const i=e[t.data.type];if(!i)return;(!i.sourceOrigin||i.sourceOrigin.startsWith(t.origin))&&i.callback(t.origin,t);});}
function reopenSidebarIfOpen(e,t,i){sessionStorage.getItem("wlIsSidebarOpen")==="true"&&openSidebar(e,t,i);}
class Configuration{constructor(){(this.jsonLdDisabled=this.getSetting("data-disable-jsonld","disableJsonLd","false")==="true"),(this.sidebarUrl=this.getSetting("data-sidebar-url","sidebarUrl","https://cloud-sidebar.wordlift.io")),(this.webPageUrl=this.getSetting("data-web-page-url","webPageUrl",window.location.href)),(this.targetOrigin=this.getSetting("data-target-origin","targetOrigin",this.webPageUrl));}
getSetting(t,i,n){if(document.currentScript&&document.currentScript.hasAttribute(t)){const o=document.currentScript.getAttribute(t);if(o!==null)return o;}
return window._wlCloudSettings&&window._wlCloudSettings[i]?window._wlCloudSettings[i]:n;}
static getInstance(){return(Configuration.instance||(Configuration.instance=new Configuration()),Configuration.instance);}}
main();