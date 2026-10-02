import json, os, shutil, psutil, torch
info={"cpu_logical":psutil.cpu_count(),"cpu_physical":psutil.cpu_count(False),"ram_gib":round(psutil.virtual_memory().total/2**30,2),"disk_free_gib":round(shutil.disk_usage('.').free/2**30,2),"cuda":torch.cuda.is_available(),"gpu_count":torch.cuda.device_count()}
if info["cuda"]: info["gpus"]=[{"index":i,"name":torch.cuda.get_device_name(i),"vram_gib":round(torch.cuda.get_device_properties(i).total_memory/2**30,2)} for i in range(info["gpu_count"])]
print(json.dumps(info,indent=2)); os.makedirs('artifacts',exist_ok=True); json.dump(info,open('artifacts/resources.json','w'),indent=2)
