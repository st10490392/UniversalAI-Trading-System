// UniversalAI MT5 execution adapter. Configure ServerURL in MT5 WebRequest allow-list.
#property strict
#property version "1.00"
#property description "Polls UniversalAI brain; local drawdown protection and risk sizing."
#include <Trade/Trade.mqh>
CTrade trade;

input string ServerURL="http://127.0.0.1:8000";
input string LicenseKey="";
input int PollSeconds=2;
input long MagicNumber=20261125;
input int SlippagePoints=20;
input bool DemoXauNyBreakout=false;   // opt-in, XAUUSD on MT5 demo only
input bool DemoRollingORB=false;       // opt-in, Range Break 100 Index on MT5 demo only
input int MaxSignalAgeSeconds=1800;   // reject signals whose M15 bar is older than this (server time)

#define HARD_DAILY_DD_PCT 5.0
#define HARD_TOTAL_DD_PCT 10.0
#define HARD_RISK_PCT 5.0

double max_daily_dd_pct=HARD_DAILY_DD_PCT,max_total_dd_pct=HARD_TOTAL_DD_PCT,risk_pct=0.5;
double daily_start_equity=0.0,high_water_equity=0.0;
datetime current_server_day=0;
bool circuit_open=false;
bool daily_lock=false,total_lock=false;   // total_lock never auto-clears: delete GV *_tlock to re-arm manually
double p_high=-1.0,p_daily=-1.0;bool p_dl=false,p_tl=false;datetime last_state_save=0;
bool authorization_failed=false;
bool authorization_alerted=false;
bool poll_authorized=false;
string last_signal_id="";
string pending_signal="";
string pending_signal_id="";
double last_signal_epoch=0.0;
datetime pending_after=0;
datetime last_transport_log=0;

string SignalStateKey()
{ return "UAI_"+(string)AccountInfoInteger(ACCOUNT_LOGIN)+"_"+(string)MagicNumber+"_"+StringSubstr(_Symbol,0,20); }
string AccountStateKey(const string suffix)
{ return "UAI_"+(string)AccountInfoInteger(ACCOUNT_LOGIN)+"_"+(string)MagicNumber+"_"+suffix; }
double DayStamp()
{ MqlDateTime x;TimeToStruct(TimeCurrent(),x);return (double)(x.year*10000+x.mon*100+x.day); }
// signal_id is "<symbol>-<bar_time>-<Strategy_Name>": the old "text after last dash" parse returned 0 for every
// real id, so nothing was persisted and a restart re-fired the same bar. The server now sends numeric bar_time.
bool AlreadyHandled(const string sid,const double bar)
{ return sid==last_signal_id || (bar>0.0 && bar<=last_signal_epoch); }
void RememberSignal(const string sid,const double bar)
{ last_signal_id=sid;if(bar>last_signal_epoch)last_signal_epoch=bar;GlobalVariableSet(SignalStateKey(),last_signal_epoch); }
bool HasOpenPosition()
{
   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong t=PositionGetTicket(i);if(t==0)continue;
      if(PositionGetString(POSITION_SYMBOL)==_Symbol)return true;
   }
   return false;
}

string JsonString(const string json,const string key)
{
   string needle="\""+key+"\""; int p=StringFind(json,needle); if(p<0)return "";
   p=StringFind(json,":",p+StringLen(needle)); if(p<0)return ""; p++;
   while(p<StringLen(json) && (StringGetCharacter(json,p)==' ' || StringGetCharacter(json,p)=='\t'))p++;
   if(StringGetCharacter(json,p)=='\"'){int e=StringFind(json,"\"",p+1);if(e<0)return "";return StringSubstr(json,p+1,e-p-1);}
   int e=p;while(e<StringLen(json) && StringGetCharacter(json,e)!=',' && StringGetCharacter(json,e)!='}' && StringGetCharacter(json,e)!=' ')e++;
   return StringSubstr(json,p,e-p);
}
double JsonNumber(const string json,const string key,const double fallback)
{ string s=JsonString(json,key); if(s=="")return fallback; return StringToDouble(s); }

string PositionRiskKey(const ulong ticket,const string suffix)
{ return "UAI_"+(string)AccountInfoInteger(ACCOUNT_LOGIN)+"_"+(string)MagicNumber+"_"+(string)ticket+"_"+suffix; }

datetime DayStart(datetime t)
{ MqlDateTime x;TimeToStruct(t,x);x.hour=0;x.min=0;x.sec=0;return StructToTime(x); }

void CloseMagicPositions()
{
   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong ticket=PositionGetTicket(i);if(ticket==0)continue;
      if(PositionGetInteger(POSITION_MAGIC)==MagicNumber &&
         (!(DemoRollingORB || DemoXauNyBreakout) || PositionGetString(POSITION_SYMBOL)==_Symbol))
         trade.PositionClose(ticket);
   }
}

// Persist to BOTH terminal globals and a COMMON-folder file: globals are only flushed on clean shutdown,
// so a terminal crash/kill would otherwise lose peak equity and the lock flags.
void PersistIfNeeded(const bool force)
{
   datetime now=TimeLocal();
   if(!force && now-last_state_save<5)return;
   if(!force && high_water_equity==p_high && daily_start_equity==p_daily && daily_lock==p_dl && total_lock==p_tl)return;
   GlobalVariableSet(AccountStateKey("day"),DayStamp());
   GlobalVariableSet(AccountStateKey("daily"),daily_start_equity);
   GlobalVariableSet(AccountStateKey("high"),high_water_equity);
   GlobalVariableSet(AccountStateKey("lock"),daily_lock?1.0:0.0);
   GlobalVariableSet(AccountStateKey("tlock"),total_lock?1.0:0.0);
   int h=FileOpen(AccountStateKey("state")+".txt",FILE_WRITE|FILE_TXT|FILE_ANSI|FILE_COMMON);
   if(h!=INVALID_HANDLE)
   {
      FileWriteString(h,StringFormat("%.0f|%.2f|%.2f|%d|%d",DayStamp(),daily_start_equity,high_water_equity,daily_lock?1:0,total_lock?1:0));
      FileClose(h);
   }
   p_high=high_water_equity;p_daily=daily_start_equity;p_dl=daily_lock;p_tl=total_lock;last_state_save=now;
}

// Conservative merge of globals + file: highest peak, highest same-day start equity, any lock wins.
void LoadState(const double equity)
{
   double stamp=DayStamp(),hw=0.0,daily=0.0;bool dl=false,tl=false;
   if(GlobalVariableCheck(AccountStateKey("high")))hw=GlobalVariableGet(AccountStateKey("high"));
   if(GlobalVariableCheck(AccountStateKey("tlock")) && GlobalVariableGet(AccountStateKey("tlock"))>0.5)tl=true;
   if(GlobalVariableCheck(AccountStateKey("day")) && GlobalVariableGet(AccountStateKey("day"))==stamp && GlobalVariableCheck(AccountStateKey("daily")))
   {
      daily=GlobalVariableGet(AccountStateKey("daily"));
      if(GlobalVariableCheck(AccountStateKey("lock")) && GlobalVariableGet(AccountStateKey("lock"))>0.5)dl=true;
   }
   int h=FileOpen(AccountStateKey("state")+".txt",FILE_READ|FILE_TXT|FILE_ANSI|FILE_COMMON);
   if(h!=INVALID_HANDLE)
   {
      string line=FileReadString(h);FileClose(h);string p[];
      if(StringSplit(line,'|',p)==5)
      {
         hw=MathMax(hw,StringToDouble(p[2]));
         if(StringToDouble(p[4])>0.5)tl=true;
         if(StringToDouble(p[0])==stamp){daily=MathMax(daily,StringToDouble(p[1]));if(StringToDouble(p[3])>0.5)dl=true;}
      }
   }
   high_water_equity=MathMax(hw,equity);
   daily_start_equity=daily>0.0?daily:equity;   // equity may be 0 before the first account update; CheckCircuit re-seeds
   daily_lock=dl;total_lock=tl;
}

void CheckCircuit()
{
   if(!DemoOrbAccountAllowed()){circuit_open=true;return;}
   double equity=AccountInfoDouble(ACCOUNT_EQUITY);
   if(equity<=0.0 || !MathIsValidNumber(equity)){circuit_open=true;return;}   // unknown equity: block entries, never trade blind
   datetime day=DayStart(TimeCurrent());
   if(day!=current_server_day){current_server_day=day;daily_start_equity=equity;daily_lock=false;PersistIfNeeded(true);}
   if(daily_start_equity<=0.0)daily_start_equity=equity;   // old code left 0 here => daily check silently disabled all day
   if(high_water_equity<=0.0)high_water_equity=equity;
   if(equity>high_water_equity)high_water_equity=equity;
   // ACCOUNT_EQUITY already includes floating P/L, so intra-trade breaches trip here on every tick and timer.
   double daily_dd=(daily_start_equity-equity)/daily_start_equity*100.0;
   double total_dd=(high_water_equity-equity)/high_water_equity*100.0;
   if(daily_dd>=max_daily_dd_pct)daily_lock=true;
   if(total_dd>=max_total_dd_pct)total_lock=true;
   circuit_open=daily_lock||total_lock;
   PersistIfNeeded(daily_lock!=p_dl || total_lock!=p_tl);
   if(circuit_open)CloseMagicPositions();
}

int StepDigits(const double v)
{ int d=0;double s=v;while(d<8 && MathAbs(s-MathRound(s))>1e-9){s*=10.0;d++;}return d; }

// Returns 0 (skip trade) instead of rounding UP to SYMBOL_VOLUME_MIN: the old MathMax(minv,..) silently
// exceeded the risk budget whenever risk-sized lots < min lot (Vol 75, Range Break, small accounts).
double NormalizeLots(const double raw)
{
   double minv=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN),maxv=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MAX),step=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP);
   if(step<=0.0 || minv<=0.0 || maxv<minv || !MathIsValidNumber(raw) || raw<minv)return 0.0;
   double lots=MathMin(maxv,raw);
   double limit=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_LIMIT);
   if(limit>0.0)
   {
      double held=0.0;
      for(int i=PositionsTotal()-1;i>=0;i--){ulong t=PositionGetTicket(i);if(t!=0 && PositionGetString(POSITION_SYMBOL)==_Symbol)held+=PositionGetDouble(POSITION_VOLUME);}
      lots=MathMin(lots,limit-held);if(lots<minv)return 0.0;
   }
   lots=minv+MathFloor((lots-minv)/step+1e-9)*step;   // step grid is anchored at min volume
   if(lots<minv || lots>maxv)return 0.0;
   return NormalizeDouble(lots,MathMax(StepDigits(step),StepDigits(minv)));
}

// lots=0 => skip. risk_used_pct is the real post-rounding loss at SL as % of min(balance,equity).
double RiskLots(const ENUM_ORDER_TYPE type,const double entry,const double sl,double &risk_used_pct)
{
   risk_used_pct=0.0;
   double base=MathMin(AccountInfoDouble(ACCOUNT_BALANCE),AccountInfoDouble(ACCOUNT_EQUITY));
   double rp=MathMin(risk_pct,HARD_RISK_PCT);
   if(base<=0.0 || rp<=0.0 || !MathIsValidNumber(rp))return 0.0;
   double risk_money=base*rp/100.0,profit=0.0,loss=0.0;
   if(OrderCalcProfit(type,_Symbol,1.0,entry,sl,profit))loss=MathAbs(profit);
   if(loss<=0.0)   // fallback: tick math (guard zero tick size/value on exotic symbols)
   {
      double tv=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_VALUE_LOSS),ts=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_SIZE);
      if(tv<=0.0)tv=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_VALUE);
      if(tv<=0.0 || ts<=0.0)return 0.0;
      loss=MathAbs(entry-sl)/ts*tv;
   }
   if(loss<=0.0 || !MathIsValidNumber(loss))return 0.0;
   double lots=NormalizeLots(risk_money/loss);if(lots<=0.0)return 0.0;
   double margin=0.0;
   if(!OrderCalcMargin(type,_Symbol,lots,entry,margin) || margin<=0.0 || margin>AccountInfoDouble(ACCOUNT_MARGIN_FREE)*0.9)return 0.0;
   risk_used_pct=loss*lots/base*100.0;
   return lots;
}

void ManageBreakeven()
{
   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong ticket=PositionGetTicket(i);if(ticket==0)continue;
      if(PositionGetInteger(POSITION_MAGIC)!=MagicNumber || PositionGetString(POSITION_SYMBOL)!=_Symbol)continue;
      string risk_key=PositionRiskKey(ticket,"risk"),trigger_key=PositionRiskKey(ticket,"be");
      if(!GlobalVariableCheck(risk_key) || !GlobalVariableCheck(trigger_key))continue;
      double risk=GlobalVariableGet(risk_key),trigger=GlobalVariableGet(trigger_key);
      if(risk<=0.0 || trigger<=0.0)continue;
      ENUM_POSITION_TYPE side=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      double entry=PositionGetDouble(POSITION_PRICE_OPEN),sl=PositionGetDouble(POSITION_SL),tp=PositionGetDouble(POSITION_TP);
      double market=(side==POSITION_TYPE_BUY)?SymbolInfoDouble(_Symbol,SYMBOL_BID):SymbolInfoDouble(_Symbol,SYMBOL_ASK);
      bool reached=(side==POSITION_TYPE_BUY)?market-entry>=trigger*risk:entry-market>=trigger*risk;
      bool already=(side==POSITION_TYPE_BUY)?sl>=entry:sl>0.0 && sl<=entry;
      double min_stop=(double)SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL)*_Point;
      if(!reached || already || MathAbs(market-entry)<min_stop)continue;
      int digits=(int)SymbolInfoInteger(_Symbol,SYMBOL_DIGITS);
      if(!trade.PositionModify(ticket,NormalizeDouble(entry,digits),tp))
         Print("UniversalAI breakeven modify failed ticket=",ticket," retcode=",trade.ResultRetcode());
   }
}

bool DemoOrbAccountAllowed()
{
   if(DemoXauNyBreakout)
      return !DemoRollingORB && AccountInfoInteger(ACCOUNT_TRADE_MODE)==ACCOUNT_TRADE_MODE_DEMO && _Symbol=="XAUUSD";
   return !DemoRollingORB ||
          (AccountInfoInteger(ACCOUNT_TRADE_MODE)==ACCOUNT_TRADE_MODE_DEMO && _Symbol=="Range Break 100 Index");
}

bool SignalProfileAllowed(const string json)
{
   string profile=JsonString(json,"execution_profile");
   if(DemoXauNyBreakout)
   {
      double risk=JsonNumber(json,"risk_pct",0.0);
      return DemoOrbAccountAllowed() && profile=="demo_xau_ny_breakout" && risk>=0.25 && risk<=0.50;
   }
   if(!DemoRollingORB)return profile=="";
   return DemoOrbAccountAllowed() && profile=="demo_rolling_orb" && JsonNumber(json,"risk_pct",0.0)==0.25;
}

bool SendPoll(string &response)
{
   if(!DemoOrbAccountAllowed())return false;
   long login=AccountInfoInteger(ACCOUNT_LOGIN);
   string safe_symbol=_Symbol;StringReplace(safe_symbol,"%","%25");StringReplace(safe_symbol," ","%20");StringReplace(safe_symbol,"#","%23");
   string url=ServerURL+"/api/ea/poll/"+(string)login+"/"+safe_symbol;
   if(DemoXauNyBreakout)url+="?demo_xau_ny_breakout=true";
   if(DemoRollingORB)url+="?demo_orb_profile=true";
   char data[],result[];
   string request_headers="X-License-Key: "+LicenseKey+"\r\n"+
      "X-Account-Balance: "+DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE),2)+"\r\n"+
      "X-Account-Equity: "+DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY),2)+"\r\n";
   string response_headers="";
   ResetLastError();int code=WebRequest("GET",url,request_headers,5000,data,result,response_headers);
   if(code<200 || code>=300)
   {
      int err=GetLastError();datetime now=TimeLocal();
      if(last_transport_log==0 || now-last_transport_log>=60)
      {
         if(code==-1 && err==4014)Print("UniversalAI WebRequest is not allowed in this context. Add ",ServerURL," to MT5 Tools > Options > Expert Advisors, and run this as an EA outside Strategy Tester.");
         else Print("UniversalAI poll transport failed HTTP=",code," err=",err," url=",ServerURL);
         last_transport_log=now;
      }
      return false;
   }
   last_transport_log=0;
   response=CharArrayToString(result,0,-1,CP_UTF8);return true;
}

void HandleSignal(const string json)
{
   if(authorization_failed)return;
   string decision=JsonString(json,"decision");if(decision!="BUY" && decision!="SELL"){pending_signal="";pending_signal_id="";return;}
   if(!SignalProfileAllowed(json))return;
   string sid=JsonString(json,"signal_id");if(sid=="" || AlreadyHandled(sid,JsonNumber(json,"bar_time",0.0)) || sid==pending_signal_id)return;
   double confidence=JsonNumber(json,"confidence",0.0);if(confidence<0.75)return;
   double rp=JsonNumber(json,"risk_pct",0.0);if(rp<=0.0)return;   // unparsable risk must not fall through to min-lot sizing
   risk_pct=MathMin(rp,HARD_RISK_PCT);
   double entry=JsonNumber(json,"entry",0.0),sl=JsonNumber(json,"sl",0.0),tp=JsonNumber(json,"tp",0.0);
   double offset=JsonNumber(json,"tp_offset_points",0.0);if(offset>0.0)tp+=(decision=="BUY"?1.0:-1.0)*offset*_Point;
   if(entry<=0 || sl<=0 || tp<=0)return;
   if(decision=="BUY" && !(sl<entry && tp>entry))return;
   if(decision=="SELL" && !(tp<entry && sl>entry))return;
   double delay=JsonNumber(json,"execution_delay_seconds",0.0);
   if(delay>0){pending_signal=json;pending_signal_id=sid;pending_after=TimeCurrent()+(int)delay;return;}
   ExecuteSignal(json);
}

void ExecuteSignal(const string json)
{
   if(!SignalProfileAllowed(json))return;
   CheckCircuit();   // re-check after the (up to 15 s) prop delay and the blocking WebRequest
   if(circuit_open || authorization_failed)return;
   string decision=JsonString(json,"decision"),sid=JsonString(json,"signal_id");
   if(decision!="BUY" && decision!="SELL")return;
   double bar=JsonNumber(json,"bar_time",0.0);
   if(sid=="" || AlreadyHandled(sid,bar))return;
   if(bar<=0.0 || (double)(DemoXauNyBreakout?TimeGMT():TimeCurrent())-bar<900 || (double)(DemoXauNyBreakout?TimeGMT():TimeCurrent())-bar>=MathMin(MaxSignalAgeSeconds,1800)){Print("Stale signal skipped ",sid);RememberSignal(sid,bar);return;}
   if(HasOpenPosition()){Print("Position already open on ",_Symbol,"; skipped ",sid);RememberSignal(sid,bar);return;}
   if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED) || !MQLInfoInteger(MQL_TRADE_ALLOWED) || SymbolInfoInteger(_Symbol,SYMBOL_TRADE_MODE)!=SYMBOL_TRADE_MODE_FULL)return;
   double sl=JsonNumber(json,"sl",0),tp=JsonNumber(json,"tp",0);
   double rp=JsonNumber(json,"risk_pct",0.0);if(rp<=0.0)return;risk_pct=MathMin(rp,HARD_RISK_PCT);
   double offset=JsonNumber(json,"tp_offset_points",0);if(offset>0)tp+=(decision=="BUY"?1.0:-1.0)*offset*_Point;
   double entry=(decision=="BUY")?SymbolInfoDouble(_Symbol,SYMBOL_ASK):SymbolInfoDouble(_Symbol,SYMBOL_BID);
   if(entry<=0.0)return;
   int digits=(int)SymbolInfoInteger(_Symbol,SYMBOL_DIGITS);sl=NormalizeDouble(sl,digits);tp=NormalizeDouble(tp,digits);
   double risk_distance=MathAbs(entry-sl),reward_distance=MathAbs(tp-entry);
   bool valid=(decision=="BUY")?(sl<entry && tp>entry):(tp<entry && sl>entry);
   double min_stop=(double)SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL)*_Point;
   if(sl<=0 || tp<=0 || !MathIsValidNumber(sl) || !MathIsValidNumber(tp) || !valid || risk_distance<min_stop || reward_distance<min_stop || reward_distance/risk_distance<(DemoXauNyBreakout?3.0-1e-9:2.5))
   { Print("Signal became invalid at market price; skipped ",sid);RememberSignal(sid,bar);return; }
   ENUM_ORDER_TYPE type=(decision=="BUY"?ORDER_TYPE_BUY:ORDER_TYPE_SELL);
   double used=0.0,lots=RiskLots(type,entry,sl,used);
   if(lots<=0.0){Print("Risk lot calculation failed / below min lot / margin; skipped ",sid);RememberSignal(sid,bar);return;}
   // Do not open a trade whose stop-out alone would breach a limit given current drawdown.
   double eq=AccountInfoDouble(ACCOUNT_EQUITY);
   double dnow=(daily_start_equity-eq)/daily_start_equity*100.0,tnow=(high_water_equity-eq)/high_water_equity*100.0;
   if(dnow+used>=max_daily_dd_pct || tnow+used>=max_total_dd_pct){Print("DD headroom too small for risk ",DoubleToString(used,2),"%; skipped ",sid);RememberSignal(sid,bar);return;}
   trade.SetExpertMagicNumber(MagicNumber);trade.SetDeviationInPoints(SlippagePoints);trade.SetTypeFillingBySymbol(_Symbol);
   if(!DemoOrbAccountAllowed())return;
   if(DemoXauNyBreakout)
   {
      datetime utc=TimeGMT();
      MqlDateTime clock;TimeToStruct(utc,clock);
      if(clock.day_of_week==0 || clock.day_of_week==6 || clock.hour<13 || clock.hour>=16 ||
         utc-bar<900 || utc-bar>=1800)return;
      string session_key="UAI_XAU_NY_"+(string)AccountInfoInteger(ACCOUNT_LOGIN);
      if(!GlobalVariableCheck(session_key))return;
      double previous=GlobalVariableGet(session_key), session=MathFloor((double)utc/86400.0);
      if(previous>=session || !GlobalVariableSetOnCondition(session_key,session,previous))return;
      GlobalVariablesFlush();
   }
   RememberSignal(sid,bar);   // at-most-once: persist BEFORE sending so a timeout/crash cannot re-fire the same signal
   bool ok=(decision=="BUY")?trade.Buy(lots,_Symbol,0.0,sl,tp,"UniversalAI "+sid):trade.Sell(lots,_Symbol,0.0,sl,tp,"UniversalAI "+sid);
   if(ok)
   {
      double be=JsonNumber(json,"breakeven_at_r",0.0);
      if(be>0.0)
      {
         for(int i=PositionsTotal()-1;i>=0;i--)
         {
            ulong ticket=PositionGetTicket(i);if(ticket==0)continue;
            if(PositionGetInteger(POSITION_MAGIC)!=MagicNumber || PositionGetString(POSITION_SYMBOL)!=_Symbol)continue;
            double position_entry=PositionGetDouble(POSITION_PRICE_OPEN),position_sl=PositionGetDouble(POSITION_SL);
            string risk_key=PositionRiskKey(ticket,"risk"),trigger_key=PositionRiskKey(ticket,"be");
            if(!GlobalVariableCheck(risk_key))GlobalVariableSet(risk_key,MathAbs(position_entry-position_sl));
            GlobalVariableSet(trigger_key,be);
         }
      }
      pending_signal_id="";
   }
   else Print("Order failed ",sid," retcode=",trade.ResultRetcode()," ",trade.ResultRetcodeDescription()," (not retried this bar)");
}

void Poll()
{
   poll_authorized=false;
   string json;if(!SendPoll(json))return;
   if(StringFind(json,"\"command\":\"UNAUTHORIZED\"")>=0 || StringFind(json,"\"command\": \"UNAUTHORIZED\"")>=0)
   { authorization_failed=true;pending_signal="";pending_signal_id="";if(!authorization_alerted){Alert("UniversalAI: LICENSE UNAUTHORIZED for this MT5 account. Trading halted.");Print("UniversalAI: server rejected license/account binding; execution is disabled until authorization returns.");authorization_alerted=true;}return; }
   if(authorization_failed){authorization_failed=false;authorization_alerted=false;Print("UniversalAI: license authorization restored.");}
   poll_authorized=true;
   // Server may only TIGHTEN limits (server schema allows up to 100%); bad/zero values keep the previous limit.
   double md=JsonNumber(json,"max_daily_dd_pct",0.0),mt=JsonNumber(json,"max_total_dd_pct",0.0);
   if(md>0.0)max_daily_dd_pct=MathMin(md,HARD_DAILY_DD_PCT);
   if(mt>0.0)max_total_dd_pct=MathMin(mt,HARD_TOTAL_DD_PCT);
   CheckCircuit();if(!circuit_open)HandleSignal(json);
}

int OnInit()
{
   if(PollSeconds<1 || !DemoOrbAccountAllowed())return INIT_PARAMETERS_INCORRECT;
   if(DemoXauNyBreakout && DemoOrbAccountAllowed())
   {
      string key="UAI_XAU_NY_"+(string)AccountInfoInteger(ACCOUNT_LOGIN);
      if(!GlobalVariableCheck(key))GlobalVariableSet(key,0.0);
   }
   trade.SetExpertMagicNumber(MagicNumber);trade.SetTypeFillingBySymbol(_Symbol);current_server_day=DayStart(TimeCurrent());
   LoadState(AccountInfoDouble(ACCOUNT_EQUITY));
   circuit_open=daily_lock||total_lock;
   PersistIfNeeded(true);
   if(GlobalVariableCheck(SignalStateKey()))last_signal_epoch=GlobalVariableGet(SignalStateKey());
   CheckCircuit();   // trip immediately on restart if floating loss already breaches a limit
   EventSetTimer(PollSeconds);return INIT_SUCCEEDED;
}
void OnDeinit(const int reason){if(DemoOrbAccountAllowed())PersistIfNeeded(true);EventKillTimer();}
void OnTick(){if(!DemoOrbAccountAllowed())return;CheckCircuit();ManageBreakeven();}
void OnTimer()
{
   if(DemoXauNyBreakout && DemoOrbAccountAllowed())
   {
      MqlDateTime utc;TimeToStruct(TimeGMT(),utc);
      if(utc.hour>=21 || utc.hour<13)
         for(int i=PositionsTotal()-1;i>=0;i--)
         {
            ulong ticket=PositionGetTicket(i);
            if(ticket!=0 && PositionGetString(POSITION_SYMBOL)==_Symbol &&
               PositionGetInteger(POSITION_MAGIC)==MagicNumber &&
               StringFind(PositionGetString(POSITION_COMMENT),"UniversalAI XAUUSD-")==0)
               trade.PositionClose(ticket);
         }
   }
   if(!DemoOrbAccountAllowed())return;
   CheckCircuit();
   Poll();
   if(poll_authorized && pending_signal!="" && TimeCurrent()>=pending_after){string p=pending_signal;pending_signal="";ExecuteSignal(p);}
}
