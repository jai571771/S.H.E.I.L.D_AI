# Target Leakage Removal Report

The following variables were removed from the AI training feature set because they introduce direct mathematical target leakage or represent non-predictive high-cardinality metadata:

| Removed Column | Leakage Category | Engineering Explanation |
| :--- | :--- | :--- |
| `Future_Risk_Level` | Target | The classification target itself. |
| `Safety_Index` | Direct Class Boundary | Derived in the Rule Engine and directly thresholds risk levels (Low/Med/High/Crit). |
| `Compound_Risk_Score` | Derived Target | Synthesized max-weighted sensor risk deviation score representing safety health. |
| `Event_ID` | High Cardinality Metadata | Unique event trace identifier string; has no generalizable predictive value. |
| `Rule_R001` | Active Safety Rule | Evaluates to 1 when specific safety boundaries are exceeded, directly leaking the state severity. |
| `Rule_R002` | Active Safety Rule | Evaluates to 1 when specific safety boundaries are exceeded, directly leaking the state severity. |
| `Rule_R003` | Active Safety Rule | Evaluates to 1 when specific safety boundaries are exceeded, directly leaking the state severity. |
| `Rule_R004` | Active Safety Rule | Evaluates to 1 when specific safety boundaries are exceeded, directly leaking the state severity. |
| `Rule_R005` | Active Safety Rule | Evaluates to 1 when specific safety boundaries are exceeded, directly leaking the state severity. |
| `Rule_R006` | Active Safety Rule | Evaluates to 1 when specific safety boundaries are exceeded, directly leaking the state severity. |
| `Rule_R007` | Active Safety Rule | Evaluates to 1 when specific safety boundaries are exceeded, directly leaking the state severity. |
| `Rule_R008` | Active Safety Rule | Evaluates to 1 when specific safety boundaries are exceeded, directly leaking the state severity. |
| `Rule_R009` | Active Safety Rule | Evaluates to 1 when specific safety boundaries are exceeded, directly leaking the state severity. |
| `Rule_R010` | Active Safety Rule | Evaluates to 1 when specific safety boundaries are exceeded, directly leaking the state severity. |
| `Rule_R011` | Active Safety Rule | Evaluates to 1 when specific safety boundaries are exceeded, directly leaking the state severity. |
| `Rule_R012` | Active Safety Rule | Evaluates to 1 when specific safety boundaries are exceeded, directly leaking the state severity. |
| `Rule_R013` | Active Safety Rule | Evaluates to 1 when specific safety boundaries are exceeded, directly leaking the state severity. |
| `Rule_R014` | Active Safety Rule | Evaluates to 1 when specific safety boundaries are exceeded, directly leaking the state severity. |
| `Rule_R015` | Active Safety Rule | Evaluates to 1 when specific safety boundaries are exceeded, directly leaking the state severity. |
| `Rule_R016` | Active Safety Rule | Evaluates to 1 when specific safety boundaries are exceeded, directly leaking the state severity. |
| `Rule_R017` | Active Safety Rule | Evaluates to 1 when specific safety boundaries are exceeded, directly leaking the state severity. |
| `Rule_R018` | Active Safety Rule | Evaluates to 1 when specific safety boundaries are exceeded, directly leaking the state severity. |
| `Rule_R019` | Active Safety Rule | Evaluates to 1 when specific safety boundaries are exceeded, directly leaking the state severity. |
| `Rule_R020` | Active Safety Rule | Evaluates to 1 when specific safety boundaries are exceeded, directly leaking the state severity. |