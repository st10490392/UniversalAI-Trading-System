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

double max_daily_dd_pct=5.0,max_total_dd_pct=10.0,risk_pct=0.5;
double daily_start_equity=0.0,high_water_equity=0.0;
datetime current_server_day=0;
bool circuit_open=false;
bool authorization_failed=false;
bool authorization_alerted=false;
bool poll_authorized=false;
string last_signal_id="";
string pending_signal="";
string pending_signal_id="";
double last_signal_epoch=0.0;
datetime pending_after=0;

string SignalStateKey()
{ return "UAI_"+(string)AccountInfoInteger(ACCOUNT_LOGIN)+"_"+(string)MagicNumber+"_"+StringSubstr(_Symbol,0,20); }
string AccountStateKey(const string suffix)
{ return "UAI_"+(string)AccountInfoInteger(ACCOUNT_LOGIN)+"_"+(string)MagicNumber+"_"+suffix; }
double DayStamp()
{ MqlDateTime x;TimeToStruct(TimeCurrent(),x);return (double)(x.year*10000+x.mon*100+x.day); }
double SignalEpoch(const string sid)
{
   int dash=-1;for(int i=0;i<StringLen(sid);i++)if(StringGetCharacter(sid,i)=='-')dash=i;
   if(dash<0)return 0.0;return (double)StringToInteger(StringSubstr(sid,dash+1));
}
bool AlreadyHandled(const string sid)
{ double epoch=SignalEpoch(sid);return sid==last_signal_id || (epoch>0.0 && epoch<=last_signal_epoch); }
void RememberSignal(const string sid)
{ last_signal_id=sid;last_signal_epoch=MathMax(last_signal_epoch,SignalEpoch(sid));GlobalVariableSet(SignalStateKey(),last_signal_epoch); }

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

datetime DayStart(datetime t)
{ MqlDateTime x;TimeToStruct(t,x);x.hour=0;x.min=0;x.sec=0;return StructToTime(x); }

void CloseMagicPositions()
{
   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong ticket=PositionGetTicket(i);if(ticket==0)continue;
      if(PositionGetInteger(POSITION_MAGIC)==MagicNumber)
         trade.PositionClose(ticket);
   }
}

void CheckCircuit()
{
   datetime now=TimeCurrent(),day=DayStart(now);double equity=AccountInfoDouble(ACCOUNT_EQUITY);
   if(day!=current_server_day){current_server_day=day;daily_start_equity=equity;circuit_open=false;GlobalVariableSet(AccountStateKey("day"),DayStamp());GlobalVariableSet(AccountStateKey("daily"),daily_start_equity);GlobalVariableSet(AccountStateKey("lock"),0.0);}
   if(high_water_equity<=0)high_water_equity=equity;
   high_water_equity=MathMax(high_water_equity,equity);GlobalVariableSet(AccountStateKey("high"),high_water_equity);
   if(GlobalVariableCheck(AccountStateKey("lock")) && GlobalVariableGet(AccountStateKey("lock"))>0.5)circuit_open=true;
   double daily_dd=daily_start_equity>0?(daily_start_equity-equity)/daily_start_equity*100.0:0.0;
   double total_dd=high_water_equity>0?(high_water_equity-equity)/high_water_equity*100.0:0.0;
   if(daily_dd>=max_daily_dd_pct || total_dd>=max_total_dd_pct){circuit_open=true;GlobalVariableSet(AccountStateKey("lock"),1.0);}
   if(circuit_open)CloseMagicPositions();
}

double NormalizeLots(double lots)
{
   double minv=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN),maxv=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MAX),step=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP);
   if(step<=0.0)return 0.0;
   lots=MathMax(minv,MathMin(maxv,lots));
   lots=MathFloor(lots/step+1e-9)*step;
   int digits=2;if(step<0.01)digits=3;if(step<0.001)digits=4;
   return NormalizeDouble(MathMax(minv,lots),digits);
}

double RiskLots(const ENUM_ORDER_TYPE type,const double entry,const double sl)
{
   double risk_money=AccountInfoDouble(ACCOUNT_BALANCE)*risk_pct/100.0,profit=0.0;
   if(!OrderCalcProfit(type,_Symbol,1.0,entry,sl,profit))return 0.0;
   double loss=MathAbs(profit);if(loss<=0.0)return 0.0;
   return NormalizeLots(risk_money/loss);
}

bool SendPoll(string &response)
{
   long login=AccountInfoInteger(ACCOUNT_LOGIN);
   string safe_symbol=_Symbol;StringReplace(safe_symbol,"%","%25");StringReplace(safe_symbol," ","%20");StringReplace(safe_symbol,"#","%23");
   string url=ServerURL+"/api/ea/poll/"+(string)login+"/"+safe_symbol+
      "?balance="+DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE),2)+
      "&equity="+DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY),2);
   char data[],result[];string request_headers="X-License-Key: "+LicenseKey+"\r\n",response_headers="";
   ResetLastError();int code=WebRequest("GET",url,request_headers,5000,data,result,response_headers);
   if(code<200 || code>=300){Print("UniversalAI poll failed HTTP=",code," err=",GetLastError());return false;}
   response=CharArrayToString(result,0,-1,CP_UTF8);return true;
}

void HandleSignal(const string json)
{
   if(authorization_failed)return;
   string decision=JsonString(json,"decision");if(decision!="BUY" && decision!="SELL"){pending_signal="";pending_signal_id="";return;}
   string sid=JsonString(json,"signal_id");if(sid=="" || AlreadyHandled(sid) || sid==pending_signal_id)return;
   double confidence=JsonNumber(json,"confidence",0.0);if(confidence<0.75)return;
   risk_pct=JsonNumber(json,"risk_pct",risk_pct);
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
   if(circuit_open || authorization_failed)return;
   string decision=JsonString(json,"decision"),sid=JsonString(json,"signal_id");
   if(sid=="" || AlreadyHandled(sid))return;
   double sl=JsonNumber(json,"sl",0),tp=JsonNumber(json,"tp",0);
   risk_pct=JsonNumber(json,"risk_pct",risk_pct);
   double offset=JsonNumber(json,"tp_offset_points",0);if(offset>0)tp+=(decision=="BUY"?1.0:-1.0)*offset*_Point;
   double entry=(decision=="BUY")?SymbolInfoDouble(_Symbol,SYMBOL_ASK):SymbolInfoDouble(_Symbol,SYMBOL_BID);
   if(entry<=0.0)return;
   int digits=(int)SymbolInfoInteger(_Symbol,SYMBOL_DIGITS);sl=NormalizeDouble(sl,digits);tp=NormalizeDouble(tp,digits);
   double risk_distance=MathAbs(entry-sl),reward_distance=MathAbs(tp-entry);
   bool valid=(decision=="BUY")?(sl<entry && tp>entry):(tp<entry && sl>entry);
   double min_stop=(double)SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL)*_Point;
   if(!valid || risk_distance<min_stop || reward_distance<min_stop || reward_distance/risk_distance<2.5)
   { Print("Signal became invalid at market price; skipped ",sid);RememberSignal(sid);return; }
   ENUM_ORDER_TYPE type=(decision=="BUY"?ORDER_TYPE_BUY:ORDER_TYPE_SELL);
   double lots=RiskLots(type,entry,sl);if(lots<=0){Print("Risk lot calculation failed for ",sid);return;}
   trade.SetExpertMagicNumber(MagicNumber);trade.SetDeviationInPoints(SlippagePoints);
   bool ok=(decision=="BUY")?trade.Buy(lots,_Symbol,0.0,sl,tp,"UniversalAI "+sid):trade.Sell(lots,_Symbol,0.0,sl,tp,"UniversalAI "+sid);
   if(ok){RememberSignal(sid);pending_signal_id="";}
   else Print("Order failed ",sid," retcode=",trade.ResultRetcode()," ",trade.ResultRetcodeDescription());
}

void Poll()
{
   poll_authorized=false;
   string json;if(!SendPoll(json))return;
   if(StringFind(json,"\"command\":\"UNAUTHORIZED\"")>=0 || StringFind(json,"\"command\": \"UNAUTHORIZED\"")>=0)
   { authorization_failed=true;pending_signal="";pending_signal_id="";if(!authorization_alerted){Alert("UniversalAI: LICENSE UNAUTHORIZED for this MT5 account. Trading halted.");Print("UniversalAI: server rejected license/account binding; execution is disabled until authorization returns.");authorization_alerted=true;}return; }
   if(authorization_failed){authorization_failed=false;authorization_alerted=false;Print("UniversalAI: license authorization restored.");}
   poll_authorized=true;
   max_daily_dd_pct=JsonNumber(json,"max_daily_dd_pct",max_daily_dd_pct);
   max_total_dd_pct=JsonNumber(json,"max_total_dd_pct",max_total_dd_pct);
   risk_pct=JsonNumber(json,"risk_pct",risk_pct);
   CheckCircuit();if(!circuit_open)HandleSignal(json);
}

int OnInit()
{
   if(PollSeconds<1)return INIT_PARAMETERS_INCORRECT;
   trade.SetExpertMagicNumber(MagicNumber);current_server_day=DayStart(TimeCurrent());
   double equity=AccountInfoDouble(ACCOUNT_EQUITY);
   high_water_equity=GlobalVariableCheck(AccountStateKey("high"))?GlobalVariableGet(AccountStateKey("high")):equity;
   if(GlobalVariableCheck(AccountStateKey("day")) && GlobalVariableGet(AccountStateKey("day"))==DayStamp() && GlobalVariableCheck(AccountStateKey("daily")))daily_start_equity=GlobalVariableGet(AccountStateKey("daily"));
   else{daily_start_equity=equity;GlobalVariableSet(AccountStateKey("day"),DayStamp());GlobalVariableSet(AccountStateKey("daily"),daily_start_equity);GlobalVariableSet(AccountStateKey("lock"),0.0);}
   circuit_open=GlobalVariableCheck(AccountStateKey("lock")) && GlobalVariableGet(AccountStateKey("lock"))>0.5;
   if(GlobalVariableCheck(SignalStateKey()))last_signal_epoch=GlobalVariableGet(SignalStateKey());
   EventSetTimer(PollSeconds);return INIT_SUCCEEDED;
}
void OnDeinit(const int reason){EventKillTimer();}
void OnTick(){CheckCircuit();}
void OnTimer()
{
   CheckCircuit();
   Poll();
   if(poll_authorized && pending_signal!="" && TimeCurrent()>=pending_after){string p=pending_signal;pending_signal="";ExecuteSignal(p);}
}
