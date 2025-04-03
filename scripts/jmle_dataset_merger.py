from datasets import load_dataset, DatasetDict
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import json

def rematch_jmle_datasets(include_unmatched=None):
    """
    元のNMLEデータセットとCoTデータセットを問題文の類似度に基づいて再マッチングして完全なデータセットを作成する関数
    
    Args:
        include_unmatched: アンマッチのデータを含めるかどうか。None=ユーザーに尋ねる、True=含める、False=含めない
    """
    # アンマッチのデータを含めるかどうかを決定
    if include_unmatched is None:
        include_unmatched_input = input("アンマッチのデータ（CoTがないデータ）を最終的なデータセットに含めますか？ (y/n): ")
        include_unmatched = include_unmatched_input.lower() == 'y'
    
    print(f"アンマッチのデータを{'含める' if include_unmatched else '含めない'}設定で処理します。")
    print("元のNMLEデータセットをロード中...")
    original_dataset = load_dataset("longisland3/NMLE")
    
    print("CoTデータセットをロード中...")
    cot_dataset = load_dataset("doctorin/JMLE-CoT-gemini-2.5-pro-exp-03-25")
    
    print(f"元データセット: {len(original_dataset['train'])} サンプル")
    print(f"CoTデータセット: {len(cot_dataset['train'])} サンプル")
    
    # 元のデータセットをリスト化（問題文の抽出用）
    print("問題文ベクトルを作成中...")
    questions = [example['question'] for example in original_dataset['train']]
    
    # 問題文をTF-IDFベクトル化
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5))
    question_vectors = vectorizer.fit_transform(questions)
    
    # CoTデータセットからの回答と説明を抽出
    cot_data = []
    for i, example in enumerate(cot_dataset['train']):
        cot_data.append({
            'index': i,
            'answer_updated': example['answer_updated'],
            'explanation_updated': example['explanation_updated'],
            'cot': example['cot']
        })
    
    # 新しい完全なデータセットの作成
    complete_data = []
    not_matched = []
    
    # ヒューリスティックマッチングのための準備
    # 正解の出現回数をカウント
    answer_counts = {}
    for example in original_dataset['train']:
        answer_key = ''.join(sorted(example['answer']))
        if answer_key in answer_counts:
            answer_counts[answer_key] += 1
        else:
            answer_counts[answer_key] = 1
    
    # 使用済みCoTエントリを記録
    used_cot_indices = set()
    
    # マッチング状態を記録するための変数
    match_status = {}  # 問題番号と最終的なマッチング状況（'high', 'medium', 'low', 'none'）
    
    # マッチングアルゴリズム
    print("データセットのマッチング中...")
    for orig_idx, orig_example in tqdm(enumerate(original_dataset['train']), total=len(original_dataset['train']), desc="元データセットを処理中"):
        orig_answer = ''.join(sorted(orig_example['answer']))
        
        # 正解が一致するCoTエントリを取得（まだ使用されていないもの）
        matching_cots = [
            cot for cot in cot_data 
            if cot['answer_updated'] == orig_answer and cot['index'] not in used_cot_indices
        ]
        
        if matching_cots:
            # ヒューリスティックマッチング：問題文末尾の正解パターンを確認
            best_match_idx = None
            
            # 正解が複数の問題で同じ場合のみ特別処理を行う
            if answer_counts[orig_answer] > 1 and len(matching_cots) > 0:
                # 説明文中に選択肢の特徴的なキーワードがあるかチェック
                for choice_idx, choice in enumerate(orig_example['choices']):
                    choice_letter = chr(97 + choice_idx)  # a, b, c, ...
                    if choice_letter in orig_answer:
                        # この選択肢に関するキーワードが説明文に含まれているCoTを探す
                        keywords = choice.split()
                        keywords = [k for k in keywords if len(k) > 3]  # 短すぎる単語を除外
                        
                        if keywords:
                            for cot_idx, cot in enumerate(matching_cots):
                                # キーワードマッチのスコアを計算
                                score = sum(1 for keyword in keywords if keyword in cot['explanation_updated'])
                                if score > 0:
                                    best_match_idx = cot_idx
                                    break
            
            # 特別なマッチングが見つからなかった場合は最初のマッチを使用
            if best_match_idx is None and matching_cots:
                best_match_idx = 0
                
            if best_match_idx is not None:
                cot_match = matching_cots[best_match_idx]
                used_cot_indices.add(cot_match['index'])
                
                # 完全なエントリを作成
                confidence = 'high' if answer_counts[orig_answer] == 1 else 'medium'
                complete_entry = {
                    'question': orig_example['question'],
                    'choices': orig_example['choices'],
                    'answer': orig_example['answer'],
                    'explanation': cot_match['explanation_updated'],
                    'cot': cot_match['cot'],
                    'answer_updated': cot_match['answer_updated'],
                    'confidence': confidence
                }
                
                match_status[orig_idx] = confidence
                complete_data.append(complete_entry)
            else:
                not_matched.append(orig_idx)
                match_status[orig_idx] = 'none'
        else:
            not_matched.append(orig_idx)
            match_status[orig_idx] = 'none'
    
    print(f"マッチした問題: {len(complete_data)}/{len(original_dataset['train'])}")
    print(f"マッチしなかった問題: {len(not_matched)}")
    
    # マッチしなかった問題をTF-IDFベクトルの類似度で再マッチング
    if not_matched:
        print("未マッチの問題をTF-IDF類似度で再マッチング中...")
        unused_cots = [cot for cot in cot_data if cot['index'] not in used_cot_indices]
        
        if unused_cots:
            # 未使用のCoTデータから説明文を取得
            cot_explanations = [cot['explanation_updated'] for cot in unused_cots]
            
            # 説明文をベクトル化
            explanation_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5))
            explanation_vectors = explanation_vectorizer.fit_transform(cot_explanations)
            
            for orig_idx in tqdm(not_matched, desc="未マッチの問題を処理中"):
                orig_example = original_dataset['train'][orig_idx]
                orig_question = orig_example['question']
                
                # 問題文をベクトル化
                question_vector = vectorizer.transform([orig_question])
                
                # 説明文との類似度を計算
                similarities = cosine_similarity(question_vector, explanation_vectors)
                best_match_idx = np.argmax(similarities[0])
                
                if similarities[0][best_match_idx] > 0.3:  # 閾値を設定
                    cot_match = unused_cots[best_match_idx]
                    used_cot_indices.add(cot_match['index'])
                    
                    # 完全なエントリを作成
                    complete_entry = {
                        'question': orig_example['question'],
                        'choices': orig_example['choices'],
                        'answer': orig_example['answer'],
                        'explanation': cot_match['explanation_updated'],
                        'cot': cot_match['cot'],
                        'answer_updated': cot_match['answer_updated'],
                        'confidence': 'low'  # 類似度ベースの低い信頼度
                    }
                    
                    match_status[orig_idx] = 'low'
                    complete_data.append(complete_entry)
                else:
                    # マッチしないサンプルも追加（CoT情報なし）
                    complete_entry = {
                        'question': orig_example['question'],
                        'choices': orig_example['choices'],
                        'answer': orig_example['answer'],
                        'explanation': orig_example.get('explanation', ''),
                        'cot': '',
                        'answer_updated': ''.join(sorted(orig_example['answer'])),
                        'confidence': 'none'  # マッチなし
                    }
                    
                    match_status[orig_idx] = 'none'
                    complete_data.append(complete_entry)
        else:
            # 未使用のCoTがない場合は元のデータをそのまま追加
            for orig_idx in not_matched:
                orig_example = original_dataset['train'][orig_idx]
                
                complete_entry = {
                    'question': orig_example['question'],
                    'choices': orig_example['choices'],
                    'answer': orig_example['answer'],
                    'explanation': orig_example.get('explanation', ''),
                    'cot': '',
                    'answer_updated': ''.join(sorted(orig_example['answer'])),
                    'confidence': 'none'  # マッチなし
                }
                
                match_status[orig_idx] = 'none'
                complete_data.append(complete_entry)
    
    # マッチング結果を保存
    with open('matching_results.json', 'w', encoding='utf-8') as f:
        json.dump(match_status, f, ensure_ascii=False, indent=2)
    
    # アンマッチのデータを除外するか判断
    if not include_unmatched:
        # 信頼度が'none'ではないデータのみを残す
        complete_data = [entry for entry in complete_data if entry['confidence'] != 'none']
        print(f"アンマッチのデータを除外し、{len(complete_data)}サンプルになりました。")
    
    # DataFrameに変換
    complete_df = pd.DataFrame(complete_data)
    
    # 信頼度の統計
    confidence_stats = complete_df['confidence'].value_counts()
    print("\n信頼度の統計:")
    for conf, count in confidence_stats.items():
        print(f"{conf}: {count} サンプル")
    
    # アンマッチだった問題番号を一覧表示
    unmatched_indices = [idx for idx, status in match_status.items() if status == 'none']
    print(f"\nアンマッチだった問題は {len(unmatched_indices)} 件です")
    print("アンマッチの問題番号リスト:")
    print(unmatched_indices)
    
    # アンマッチの問題番号をCSVで保存
    unmatched_df = pd.DataFrame({'problem_index': unmatched_indices})
    unmatched_df.to_csv('unmatched_problems.csv', index=False)
    print("アンマッチの問題番号リストを 'unmatched_problems.csv' に保存しました")
    
    # DatasetDictに変換
    complete_dataset = DatasetDict({
        'train': original_dataset['train'].from_pandas(complete_df, preserve_index=False)
    })
    
    print("\n完全なデータセットの統計情報:")
    print(f"サンプル数: {len(complete_dataset['train'])}")
    print(f"カラム一覧: {complete_dataset['train'].column_names}")
    
    # ローカルに保存するか尋ねる
    save_local = input("完全なデータセットをローカルに保存しますか？ (y/n): ")
    if save_local.lower() == 'y':
        output_dir = "complete_jmle_dataset_improved"
        os.makedirs(output_dir, exist_ok=True)
        complete_dataset.save_to_disk(output_dir)
        print(f"データセットを {output_dir} に保存しました")
    
    # HuggingFaceに保存するか尋ねる
    push_to_hub = input("完全なデータセットをHugging Faceにアップロードしますか？ (y/n): ")
    if push_to_hub.lower() == 'y':
        repo_name = input("リポジトリ名を入力してください (例: your-username/JMLE-Complete): ")
        private = input("プライベートリポジトリにしますか？ (y/n): ").lower() == 'y'
        complete_dataset.push_to_hub(repo_name, private=private)
        print(f"データセットを {repo_name} にアップロードしました")
    
    return complete_dataset, unmatched_indices

def analyze_unmatched_problems(original_dataset, unmatched_indices):
    """
    アンマッチだった問題の分析を行う関数
    """
    print("\nアンマッチだった問題の分析:")
    
    # アンマッチ問題のサンプルを表示
    num_samples = min(5, len(unmatched_indices))
    if num_samples > 0:
        print(f"\n最初の{num_samples}件のアンマッチ問題サンプル:")
        
        for i, idx in enumerate(unmatched_indices[:num_samples]):
            sample = original_dataset['train'][idx]
            print(f"\nサンプル {idx} (サンプル {i+1}/{num_samples})")
            print("="*50)
            print("問題:")
            print(sample['question'])
            print("-"*30)
            print("選択肢:")
            print("\n".join(sample['choices']))
            print("-"*30)
            print("正解:")
            answer_letters = ''.join(sorted(sample['answer']))
            # 選択肢のインデックス範囲をチェックする安全なバージョン
            answer_texts = []
            for c in answer_letters:
                idx = ord(c) - 97  # 'a'のASCIIコード97を引く
                if 0 <= idx < len(sample['choices']):
                    answer_texts.append(sample['choices'][idx])
                else:
                    answer_texts.append(f"[選択肢{c}は範囲外]")
            print(f"{answer_letters} - {', '.join(answer_texts)}")
            print("="*50)
    
    # アンマッチ問題の統計
    print(f"\nアンマッチ問題の総数: {len(unmatched_indices)}")
    
    # アンマッチ問題の問題番号のCSVを読み込む方法の説明
    print("\nアンマッチ問題の詳細な分析:")
    print("'unmatched_problems.csv' ファイルには、アンマッチだった問題の番号が保存されています。")
    print("このファイルを使用して、元のデータセットからアンマッチだった問題を詳細に分析できます。")

if __name__ == "__main__":
    from dotenv import load_dotenv
    from huggingface_hub import HfApi, HfFolder
    from huggingface_hub import login as hf_login
    load_dotenv(override=True)
    hf_token = os.getenv('HF_TOKEN')
    hf_login(hf_token)
    token = HfFolder.get_token()
    api = HfApi()
    user_info = api.whoami(token=token)
    print(user_info)

    try:
        # データセットの再マッチング（アンマッチデータを含めるかどうかはユーザーに尋ねる）
        complete_dataset, unmatched_indices = rematch_jmle_datasets(include_unmatched=None)
        
        # アンマッチ問題の分析
        print("\nアンマッチ問題の分析を行いますか？ (y/n): ")
        analyze = input()
        if analyze.lower() == 'y':
            original_dataset = load_dataset("longisland3/NMLE")
            analyze_unmatched_problems(original_dataset, unmatched_indices)
            
    except Exception as e:
        import traceback
        print(f"エラーが発生しました: {e}")
        print(traceback.format_exc())