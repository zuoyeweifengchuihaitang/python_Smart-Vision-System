# -*- coding: utf-8 -*-
import os
import pickle
import cv2
from core.recognition import extract_embeddings
from config import DB_PATH, FACES_DIR

def build_face_db(faces_dir=FACES_DIR):
    face_db = {}
    blacklist = set()
    whitelist = set()

    for subdir, label_set in [('black', blacklist), ('white', whitelist)]:
        path = os.path.join(faces_dir, subdir)
        if not os.path.exists(path): continue

        for filename in os.listdir(path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(path, filename)
                print(f"[处理中] {filename}")

                try:
                    embs = extract_embeddings(full_path)
                    if embs:
                        # 选最大脸
                        best_face = max(embs, key=lambda x: (x[1][2]-x[1][0]) * (x[1][3]-x[1][1]))
                        emb = best_face[0]
                        person_id = os.path.splitext(filename)[0]
                        face_db[person_id] = emb
                        label_set.add(person_id)
                        print(f"  → 成功录入: {person_id} ({subdir})")
                    else:
                        print(f"  → [跳过] 未检测到人脸: {filename}")
                except Exception as e:
                    print(f"  → [异常] {filename}: {e}")

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with open(DB_PATH, 'wb') as f:
        pickle.dump({'embeddings': face_db, 'blacklist': blacklist, 'whitelist': whitelist}, f)

    print(f"人脸库保存到 {DB_PATH}，共 {len(face_db)} 条记录")
    return face_db, blacklist, whitelist

def load_face_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, 'rb') as f:
            data = pickle.load(f)
        print(f"加载成功：黑名单 {len(data['blacklist'])} 人，白名单 {len(data['whitelist'])} 人")
        return data['embeddings'], data['blacklist'], data['whitelist']
    return {}, set(), set()

def startup_self_check():
    """开机自检：对比文件夹图片和数据库索引"""
    print("\n" + "🔍" + " 开始开机自检...")
    folder_ids = set()
    for subdir in ['black', 'white']:
        path = os.path.join(FACES_DIR, subdir)
        if os.path.exists(path):
            files = [os.path.splitext(f)[0] for f in os.listdir(path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            folder_ids.update(files)
    
    db_embeddings, blacklist, whitelist = load_face_db()
    db_ids = set(db_embeddings.keys())
    
    if folder_ids == db_ids:
        print(f"✅ 自检通过：数据库与文件夹同步 (共 {len(folder_ids)} 人)。")
        return db_embeddings, blacklist, whitelist
    else:
        print(f"⚠️ 检测到数据不一致！正在自动重构索引...")
        return build_face_db()

def register_face(img, pid, g_type):
    embs = extract_embeddings(img)
    if not embs: return False, "未检测到人脸"
    emb, bbox, _ = max(embs, key=lambda x: (x[1][2]-x[1][0]) * (x[1][3]-x[1][1]))
    sub = 'black' if g_type == '1' else 'white'
    os.makedirs(os.path.join(FACES_DIR, sub), exist_ok=True)
    save_path = os.path.join(FACES_DIR, sub, f"{pid}.jpg")
    cv2.imencode('.jpg', img)[1].tofile(save_path)
    startup_self_check()
    return True, f"成功录入至 {sub}"

# 文件位置：database/operations.py


def delete_face(person_id):
    """
    同步数据库与文件
    """
    # 1. 加载当前数据库
    face_db, blacklist, whitelist = load_face_db()

    # 2. 无论文件还在不在，先从内存数据库中移除！
    removed_from_db = False
    if person_id in face_db:
        del face_db[person_id]
        removed_from_db = True
        print(f"🗑️ 已从内存特征库中移除: {person_id}")
    
    if person_id in blacklist: blacklist.remove(person_id)
    if person_id in whitelist: whitelist.remove(person_id)

    # 3. 尝试删除物理文件 
    # 扫描 black 和 white 两个文件夹
    for sub_dir in ['black', 'white']:
        dir_path = os.path.join(FACES_DIR, sub_dir)
        if not os.path.exists(dir_path): continue
        
        for filename in os.listdir(dir_path):
            # 只要文件名匹配 ID，统统删掉
            if os.path.splitext(filename)[0] == person_id:
                file_path = os.path.join(dir_path, filename)
                try:
                    os.remove(file_path)
                    print(f"🗑️ 物理文件已删除: {file_path}")
                except Exception as e:
                    print(f"⚠️ 物理文件删除出错 (不影响数据库清理): {e}")

    # 4. 立即保存新的 .pkl 文件
    # 这样下次重启软件时，这个人就绝对不会再出现了
    try:
        with open(DB_PATH, 'wb') as f:
            pickle.dump({'embeddings': face_db, 'blacklist': blacklist, 'whitelist': whitelist}, f)
        print("💾 数据库文件(.pkl)已强制更新")
    except Exception as e:
        return False, f"数据库保存失败: {e}"

    return True, f" 人员 [{person_id}] 已彻底移除！"